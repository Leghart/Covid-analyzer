import numpy as np
import math
import pandas as pd
import textwrap
import os
import sys
import warnings
import ssl
import smtplib
import datetime
import functools
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima import auto_arima

from data_base import init_db
from data_base import MainBase
from data_base import PredBase


# Ignore tensorflow warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Class of predicting the future of a pandemic in the selected country.
# Consists of:
# - methods of data conversion to a form correct for prediction
# - message template to be sent
# - function sending the message template
# - self implemented RBF-network (not used)
# - methods of preparing data from the database
# - functions decorators
# - ARIMA - autoregressive integrated moving average - main forecast function
# - LSTM - recurrent neural network - second forecast function
# - minor regression functions (HoltWinters, vector autoregression) (not used)
class Process:

    # Path to directory where files were placed
    path = os.path.dirname(os.path.abspath(__file__))

    # Names of columns in database.
    # Used in dynamically download data from database
    fields = ['new_cases', 'total_cases', 'total_recovered', 'active_cases',
              'new_deaths', 'total_deaths', 'tot_1M', 'fatality_ratio',
              'total_tests', 'date']

    # Constructor of downloading data from database of selected country
    def __init__(self):
        raw_data = MainBase.get_data()

        for key in __class__.fields:
            self.__dict__[key] = []

        for i in raw_data:
            for key, val in i.items():
                self.__dict__[key].append(val)

    # Method to easly-read number format (e.g. 7 521 642 instead of 7521642)
    @staticmethod
    def format_number(string):
        return " ".join(digit for digit in textwrap.wrap(
                                    str(string)[::-1], 3))[::-1]

    # Method making shift in data set - used in self implemented
    # Kohonen network
    @staticmethod
    def preprocessData(data, output, k):
        X, Y = [], []
        for i in range(len(data) - k - 1):
            x_i_mat = np.array(data[i:(i + k)])
            x_i = x_i_mat.reshape(x_i_mat.shape[0] * x_i_mat.shape[1])
            y_i = np.array(data[(i + k):(i + k + 1)][output])
            X.append(x_i)
            Y.append(y_i)
        return np.array(X), np.array(Y)

    # Method making shift in data set - used in LSTM
    @staticmethod
    def create_dataset(dataset, look_back=1):
        dataX = []
        dataY = []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # Return a special cap of data using in decorators.
    @staticmethod
    def make_cap(x_data, y_data, x_train, y_train, x_test, y_test,
                 x_pred, y_pred):
        return [[x_data, y_data], [x_train, y_train], [x_test, y_test],
                [x_pred, y_pred]]

    # Prepare message to sent, written in Polish
    def raport_to_mail(self):
        From = "Automatyczny Raport Wirusowy"
        subject = f'Raport z dnia: {self.date[-1]}'
        message = """\n

        Dzisiejsze zachorowania: {}\n
        Dzisiejsze zgony: {}\n
        ==============================================
        Wszystkie przypadki zachorowan: {}\n
        Wszystkie zgony: {}\n
        Wszyscy wyzdrowiali: {}\n
        Aktywne przypadki: {}\n
        Ilość zmarłych na 1M: {}\n
        Współczynnik smiertelnosci: {}\n
        Wszystkie wykonane testy: {}\n
        ------------ Prognoza na dzień {} ----------------
        Zachorowania: {}\n
        Zgony: {}
        """.format(
                    __class__.format_number(str(self.new_cases[-1])),
                    __class__.format_number(str(self.new_deaths[-1])),
                    __class__.format_number(str(self.total_cases[-1])),
                    __class__.format_number(str(self.total_deaths[-1])),
                    __class__.format_number(str(self.total_recovered[-1])),
                    __class__.format_number(str(self.active_cases[-1])),
                    __class__.format_number(str(self.tot_1M[-1])),
                    str(self.fatality_ratio[-1]),
                    __class__.format_number(str(self.total_cases[-1])),
                    str(self.next_day),
                    __class__.format_number(str(self.cases_pred)),
                    __class__.format_number(str(self.deaths_pred)))
        return 'Subject: {}\n\n{}'.format(subject, message.encode(
                                    'ascii', 'ignore').decode('ascii'))

    # Method of sending e-mails to list of receivers by the broadcaster
    # (e-mail) using a special browser key (each argument is in a
    # different files for privacy and enable easy extension). If you want to
    # send mail to more the one receiver, separate them using ';'
    def send_mail(self, broadcaster_handler, receiver_handler,
                  password_handler):

        port = 465
        smtp_serv = "smtp.gmail.com"
        try:
            broadcaster = open(broadcaster_handler).read()
            receiver = open(receiver_handler).read().split(';')
            password = open(password_handler).read()
            message = self.raport_to_mail()
            del receiver[-1]

            ssl_pol = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_serv, port, context=ssl_pol) as serwer:
                serwer.login(broadcaster, password)
                serwer.sendmail(broadcaster, receiver, message)
            print('Mail was sent!')
        except Exception as e:
            print('Mail sending error:', e)

    # Get only completed columns of database, returned in pandas DataFrame
    def get_data_from_self(self):
        new_data = np.arange(len(self.date))
        d = {'Date': new_data, 'Total cases': self.total_cases,
             'New cases': self.new_cases, 'Total deaths': self.total_deaths,
             'New deaths': self.new_deaths,
             'Total recovered': self.total_recovered,
             'Active cases': self.active_cases, 'Tot /1M': self.tot_1M,
             'Fatality ratio': self.fatality_ratio,
             'Total tests': self.total_tests}
        df = pd.DataFrame(data=d)
        df = df.drop(columns=['Tot /1M', 'Total tests'])
        return df

    # Self written a self-organizing map using using unsupervised learning.
    def Kohonen(self, X, klasy, alfa=0.45, il_iter=100):
        srd = sum(i for i in X) / len(X)

        X_std = np.array([(srd - X[i]) / np.linalg.norm(X[i])
                          for i in range(len(X))])
        gen = np.random.RandomState(100)
        W_rep = []
        for i in range(klasy):
            wtemp = gen.normal(loc=0.0, scale=0.01, size=len(X[0]))
            W_rep.append(wtemp / np.linalg.norm(wtemp))

        alfa_k = alfa
        W_mod = []
        for iter_k in range(il_iter):
            W_mod = []
            for i in range(len(X_std)):
                p_min = [np.linalg.norm(W_rep[j] - X_std[i])
                         for j in range(klasy)]
                min_idx = np.where(p_min == np.amin(p_min))
                W_mod.append(min_idx[0][0])
            W_mod = np.array(W_mod)

            for i in range(len(X_std)):
                W_rep[W_mod[i]] = ((W_rep[W_mod[i]] + alfa_k * (
                                   X_std[i] - W_rep[W_mod[i]])) /
                                   np.linalg.norm(W_rep[W_mod[i]]))
            alfa_k = alfa * math.exp(-1.5 * iter_k)

        return W_rep

    # Simple kind of neural network, making a approximate of function using
    # Kohonen network.
    def RBF(self, X, y, liczba_klas, scaler):
        srd = sum(i for i in X) / len(X)
        Xn = np.array([(srd - X[i]) / np.linalg.norm(X[i])
                       for i in range(len(X))])

        C = self.Kohonen(X, liczba_klas)

        N = range(len(Xn))
        S = max([np.linalg.norm(Xn[i] - Xn[j]) for i in N for j in N])

        r = S / scaler

        PHI = []
        for N in range(len(Xn)):
            pom = []
            for p in range(liczba_klas):
                odl = np.linalg.norm(Xn[N] - C[p])
                pom.append(np.exp(-np.square(odl / r)))
            PHI.append(pom)
        PHI = np.array(PHI)

        cz1 = np.linalg.pinv(np.dot(np.transpose(PHI), PHI))
        cz2 = np.dot(np.transpose(PHI), y)
        w = np.dot(cz1, cz2)

        y_rad = []
        for i in range(len(Xn)):
            tmp = 0
            for j in range(len(w)):
                tmp += (w[j] * np.exp(-np.square(np.linalg.norm(
                                                Xn[i] - C[j]) / r)))
            y_rad.append(tmp)

        return y_rad

    # Method making a figures of RBF network
    def RBF_prediction(self, keys):
        data = self.get_data_from_self()
        data = data[:-1]

        pred_dict = {}
        for i in keys:
            plt.figure(i)
            X, y = Process.preprocessData(data, i, int(len(self.date) * 0.2))
            print('Calculating prediction for {}'.format(i))
            y_rad = self.RBF(X, y, 200, 10)

            pred_dict[i] = int(y_rad[-1])

            t1 = range(len(y_rad))
            t2 = range(len(y))

            plt.plot(t1, y_rad, label="Prediction")
            plt.plot(t2, y, label="Original data")
            plt.xlabel("Days since the start of the pandemic")
            plt.legend()
            plt.title('{} ({})'.format(i, self.date[-1]))
            plt.grid()

            full_path = Process.path + r'\static\{}.png'.format(i)
            if os.path.isfile(full_path):
                os.remove(full_path)

            plt.savefig(Process.path + r'\static/{}'.format(i))
        self.cases_pred = pred_dict['New cases']
        self.deaths_pred = pred_dict['New deaths']

        self.next_day = (datetime.datetime.strptime(self.date[-1],
                         '%d.%m.%Y') +
                         datetime.timedelta(days=1)).strftime('%d.%m.%Y')

    # Function to decorate any prediction function, save plots of
    # original data, train-test set data (optional) and forecast data.
    # Before each plot save, if the figure exists, delete it.
    def plot_decorator(f):
        def func(self, *args, **kw):
            kwargs = f(self, *args)

            for key in kwargs:
                plt.figure(key)
                plt.plot(kwargs[key][0][0],
                         kwargs[key][0][1],
                         label='Original data')
                plt.plot(kwargs[key][1][0],
                         kwargs[key][1][1],
                         label='Train data')
                plt.plot(kwargs[key][2][0],
                         kwargs[key][2][1],
                         label='Test data')
                plt.plot(kwargs[key][3][0],
                         kwargs[key][3][1],
                         label='Forecast')
                plt.axvline(x=len(kwargs[key][0][0]),
                            color='k',
                            linestyle='--')
                plt.legend()
                plt.title('{} ({})'.format(key, self.date[-1]))
                plt.minorticks_on()
                plt.grid(b=True, which='minor', color='#999999',
                         linestyle='-', alpha=0.2)

                full_path = __class__.path + r'\static\{}.png'.format(key)
                if os.path.isfile(full_path):
                    os.remove(full_path)

                plt.savefig(__class__.path + r'\static/{}'.format(key))

            return kwargs
        return func

    # Function to decorate any prediction function, saves to database
    # information on the date, cases and prognosis of deaths.
    def db_decorator(f):
        def func(self, *args, **kw):
            kwargs = f(self, *args)

            cases_pred = int(kwargs['New cases'][3][1][1])
            deaths_pred = int(kwargs['New deaths'][3][1][1])

            next_day = (datetime.datetime.strptime(self.date[-1], '%d.%m.%Y') +
                        datetime.timedelta(days=1)).strftime('%d.%m.%Y')

            init_db()
            cap = {'date': next_day, 'cases_pred': cases_pred,
                   'deaths_pred': deaths_pred}
            PredBase.insert(**cap)

            return kwargs
        return func

    # Autoregressive integrated moving average - regressor making a pandemic
    # forecast. Uses limited memory BFGS optimization (lbfgs). As a parameter
    # you can provide a prognostic rate (basically uses new cases
    # and of deaths) and the number of days until the forecast is made.
    # Used decorators immediately save the result to the database and draw a
    # forecast graph and original data.
    @plot_decorator
    @db_decorator
    def ARIMA(self, keys=['New cases', 'New deaths'], days_pred=7):
        pred_dict = {}
        kwargs = {}
        for key in keys:
            data = self.get_data_from_self()[key]
            act = len(data)
            horizont = pd.DataFrame(np.zeros(days_pred))
            ndata = pd.concat([data, horizont], ignore_index=True)

            print('Calculating prediction for {}'.format(key))
            arima_model = auto_arima(data[:act], method='lbfgs')
            test = ndata[act:]

            prediction = pd.DataFrame(arima_model.predict(n_periods=len(test)),
                                      index=test.index)

            x_data = list(range(1, len(data) + 1))

            x_pred = list(range(len(data), len(data) + days_pred))
            forecast = prediction.values.flatten()

            kwargs[key] = __class__.make_cap(x_data, data,
                                             [0], [0],
                                             [0], [0],
                                             x_pred, forecast)
        return kwargs

    # Long short-term memory - artificial recurrent neural network. Split the
    # data in a 0.75-0.25 (train-test). To learn neural network, used a shift
    # horizont of 1 day in back (makes the best results). Network consists of
    # 50 units of LSTM, a fully connected layers: 1 and selected optimizer is
    # 'nadam'. Forecast was made in l following way: last downloaded data is
    # given to network (output of network is append to prediction_list,
    # where in second iteration last data will be used to feed network).
    @db_decorator
    @plot_decorator
    def LSTM(self, keys=['New cases', 'New deaths'], num_pred=7):
        pred_dict = {}
        kwargs = {}
        for key in keys:
            # Get data
            data = self.get_data_from_self()[key].values
            data = data.reshape(len(data), 1)

            # Normalize data to 0-1 (easier work with that)
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(data)

            # Set a train-test factor and split data
            train_size = int(len(dataset) * 0.75)
            test_size = len(dataset) - train_size

            train = dataset[0:train_size]
            test = dataset[train_size:len(dataset)]

            # Shift horizont
            look_back = 1
            trainX, trainY = __class__.create_dataset(train, look_back)
            testX, testY = __class__.create_dataset(test, look_back)

            # Create a special form (inputs, targets, sample_weights)
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            # Create LSTM network and learn using train-set
            model = Sequential()
            model.add(LSTM(50, input_shape=(1, look_back)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='nadam')
            model.fit(trainX, trainY, epochs=500, batch_size=64, verbose=1)

            # Use fitted network to predict a train and test sets
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)

            # Get original values
            trainPredict = scaler.inverse_transform(trainPredict)
            testPredict = scaler.inverse_transform(testPredict)

            # Main section - Forecast the next num_pred days
            prediction_list = dataset[-look_back:]
            for _ in range(num_pred):
                x = prediction_list[-look_back:]
                x = np.reshape(x, (1, -1))
                x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
                out = model.predict(x)
                prediction_list = np.append(prediction_list, out)
            prediction_list = np.reshape(prediction_list, (1, -1))
            prediction_list = scaler.inverse_transform(prediction_list)
            prediction_list = prediction_list.flatten()  # 2D -> 1D

            # Prepare veriables to plot
            bsc_time = list(range(1, len(dataset) + 1))
            prediction_dates = list(range(len(dataset),
                                    len(dataset) + num_pred + 1))
            bsc_time.extend(prediction_dates)

            trainPredictPlot = np.empty_like(dataset)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[:len(trainPredict)] = trainPredict

            testPredictPlot = np.empty_like(dataset)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(trainPredict) +
                            (look_back * 2): len(dataset) - 2] = testPredict

            kwargs[key] = __class__.make_cap(bsc_time[:len(dataset)],
                                             scaler.inverse_transform(dataset),
                                             bsc_time[:len(trainPredictPlot)],
                                             trainPredictPlot,
                                             bsc_time[:len(trainPredictPlot)],
                                             testPredictPlot,
                                             prediction_dates,
                                             prediction_list)
        return kwargs

    # Forecasting method using vector autoregression
    def VAR(self, keys=['New cases', 'New deaths']):
        from statsmodels.tsa.vector_ar.var_model import VAR
        # contrived dataset with dependency
        data1 = self.get_data_from_self()[keys[0]]
        data2 = self.get_data_from_self()[keys[1]]

        data = list()
        for i in range(len(data1)):
            v1 = data1[i]
            v2 = data2[i]

            row = [v1, v2]
            data.append(row)

        model = VAR(data)
        model_fit = model.fit()

        yhat = model_fit.forecast(model_fit.y, steps=1)
        dict = {keys[0]: int(yhat[0][0]), keys[1]: int(yhat[0][1])}
        return dict

    # Forecasting method using HoltWinters method
    def HVES(self, keys):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
        di = {}
        for k in keys:
            df = self.get_data_from_self()[k]
            df = df[10:]

            tmp = 3
            df_train = df.iloc[:-tmp]
            df_test = df.iloc[-tmp:]
            model = HWES(df_train, seasonal_periods=tmp, trend='mul',
                         seasonal='add', initialization_method='estimated')

            fitted = model.fit()
            sales_forecast = fitted.forecast(steps=tmp)

            predi = list(sales_forecast[:1])
            di[k] = int(predi[0])
        return di

# P = Process()
# P.ARIMA()

# P.PRED()
# cap = {'date': '01-01-01','cases_pred': 10,
#     'deaths_pred': 20}
# PredBase.insert(**cap)

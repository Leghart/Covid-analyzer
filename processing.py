"""
Data processing module, forecast calculation and charting.
It includes several different forecasting methods, not all of them are used,
but all of them are them have the same calling, so if you want to try one of
them just call it and print result.
"""
import numpy as np
import math
import pandas as pd
import textwrap
import os
import sys
import warnings
import ssl
import smtplib
from functools import wraps
import datetime
import functools
import inspect
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

from pmdarima.arima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_base import init_db
from data_base import MainBase
from data_base import PredBase


# Ignore tensorflow warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




class Process:
    """Class of predicting the future of a pandemic in the selected country.

    Consists of:
    - a method of changing the data format for prediction
    - message template to be sent
    - function sending the message template
    - method preparing data from the database
    - functions decorators
    - ARIMA - autoregressive integrated moving average - main forecast function
    - LSTM - recurrent neural network - second forecast function
    - minor regression functions (HoltWinters, vector autoregression)
    """

    # Path to directory where files were placed
    path = os.path.dirname(os.path.abspath(__file__))

    # Names of columns in database.
    # Used in dynamically download data from database.
    fields = ['new_cases', 'total_cases', 'total_recovered', 'active_cases',
              'new_deaths', 'total_deaths', 'tot_1M', 'fatality_ratio',
              'total_tests', 'date']

    def __init__(self):
        """
        Constructor of downloading data from database of
        selected country.

        Parameters:
        -----------
        - None

        Returns:
        --------
        - None
        """
        raw_data = MainBase.get_data()

        for key in __class__.fields:
            self.__dict__[key] = []

        for i in raw_data:
            for key, val in i.items():
                self.__dict__[key].append(val)

    @staticmethod
    def format_number(string):
        """
        Method to easly-read number format
        (e.g. 7 521 642 instead of 7521642).

        Parameters:
        -----------
        - string (string) - number to change format

        Returns:
        --------
        - (string) - changed string
        """
        return " ".join(digit for digit in textwrap.wrap(
                                    str(string)[::-1], 3))[::-1]

    @staticmethod
    def create_dataset(dataset, look_back=1):
        """
        Method making shift in data set - used in LSTM.

        Parameters:
        -----------
        - dataset (array) - vector of data
        - look_back (int) - horizont of prediction

        Returns:
        --------
        - dataX (array) - Shift x data in vector
        - dataY (array) - Shift y data in vector
        """
        dataX = []
        dataY = []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    @staticmethod
    def make_cap(x_data, y_data, x_train, y_train, x_test, y_test,
                 x_pred, y_pred):
        """
        Return a special cap of data using in decorators. If one of method
        doesnt use train-test data, will return [0, 0] for train and
        [0, 0] for test.

        Parameters:
        -----------
        - x_data (list) - x values of original data
        - y_data (list) - y values of original data
        - x_train (list) - x values of train data
        - y_train (list) - y values of train data
        - x_test (list) - x values of test data
        - y_test (list) - y values of test data
        - x_pred (list) - x values of forecast data
        - y_pred (list) - y values of forecast data

        Returns:
        --------
        - (list):
        [[x_data, y_data], [x_train, y_train],
         [x_test, y_test], [x_pred, y_pred]]
        """
        return [[x_data, y_data], [x_train, y_train], [x_test, y_test],
                [x_pred, y_pred]]

    def get_new_recovered(self):
        """
        Return the list of daily recoveries generated as difference between
        current and previous value.

        Parameters:
        -----------
        - None

        Returns:
        --------
        - new_daily_rec (list) - list of daily recovered
        """
        data_total_rec = self.total_recovered
        new_daily_rec = []
        for i in range(len(data_total_rec)):
            try:
                result = data_total_rec[i + 1] - data_total_rec[i]
                new_daily_rec.append(result)
            except IndexError:
                break

        new_daily_rec.insert(0,0) # set first value as 0
        #return pd.DataFrame(data=new_daily_rec)
        return new_daily_rec

    def raport_to_mail(self):
        """
        Prepare message to sent, written in Polish.

        Parameters:
        ----------
        - None

        Returns:
        --------
        - message (string) - correct form to send message using SMTPlib
        """
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

    def send_mail(self, broadcaster_handler, receiver_handler,
                  password_handler):
        """
        Method of sending e-mails to list of receivers by the broadcaster
        (e-mail) using a special browser key (each argument is in a
        different files for privacy and enable easy extension).

        Parameters:
        -----------
        - broadcaster_handler (string) - path to the file where the
        sender's e-mail is saved
        - receiver_handler (string) - path to the file where the receiver's
        e-mails are saved, separeted with ';'
        - password_handler (string) - path to the file, where password is saved

        Returns:
        --------
        - None.
        """
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
        except Exception as e:
            print('Mail sending error:', e)
        else:
            print('Mail was sent!')

    def get_data_from_self(self):
        """
        Selects only the completed data from an instance and
        returns it in a DataFrame.

        Parameters:
        -----------
        - keys (list) - pointers to research
        - days_pred (int) - number of days to look in the future

        Returns:
        --------
        - df (DataFrame) - completed columns
        """
        new_data = np.arange(len(self.date))
        d = {'Date': new_data, 'Total cases': self.total_cases,
             'New cases': self.new_cases, 'Total deaths': self.total_deaths,
             'New deaths': self.new_deaths,
             'Total recovered': self.total_recovered,
             'Active cases': self.active_cases, 'Tot /1M': self.tot_1M,
             'Fatality ratio': self.fatality_ratio,
             'Total tests': self.total_tests,
             'New recovered': self.get_new_recovered()}
        df = pd.DataFrame(data=d)
        df = df.drop(columns=['Tot /1M', 'Total tests'])
        return df

    @staticmethod
    def decorator_selector(pointer, *args, **kwargs):
        """

        """

        if len(args) != 0:
            dict_tmp = {'config_plot': 2,
                        'config_db': 3}
            return args[dict_tmp[pointer]]

        elif len(kwargs) != 0:
            return kwargs[pointer]
        else:
            return False



    def plot_decorator(f):
        """
        Function to decorate any prediction function, save plots of
        original data, train-test set data (optional) and forecast data.
        Before each plot save, if the figure exists, delete it.

        Parameters:
        -----------
        - None

        Returns:
        --------
        - Called function.
        """
        @wraps(f)
        def func(self, *args, **kw):
            kwargs = f(self, *args, **kw)

            config = __class__.decorator_selector('config_plot', *args, **kw)

            if config:
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
                plt.show()
                return kwargs
            else:
                return kwargs
        return func

    def db_decorator(f):
        """
        Function to decorate any prediction function, saves to database
        information on the date, cases and prognosis of deaths.

        Parameters:
        -----------
        - None

        Returns:
        --------
        - Called function.
        """
        @wraps(f)
        def func(self, *args, **kw):
            kwargs = f(self, *args, **kw)

            config = __class__.decorator_selector('config_db', *args, **kw)

            if config:
                # take only first value of forecast to save in database
                try:
                    self.cases_pred = int(kwargs['New cases'][3][1][0])
                    self.deaths_pred = int(kwargs['New deaths'][3][1][0])

                    self.next_day = ((datetime.datetime.strptime(self.date[-1],
                                      '%d.%m.%Y') + datetime.timedelta(days=1))
                                     .strftime('%d.%m.%Y'))
                except KeyError:
                    return kwargs

                init_db()
                cap = {'date': self.next_day, 'cases_pred': self.cases_pred,
                       'deaths_pred': self.deaths_pred}
                PredBase.insert(**cap)
                print('Database was updated.')
                return kwargs
            else:
                return kwargs
        return func

    @plot_decorator
    @db_decorator
    def ARIMA(self, keys=['New cases', 'New deaths'], days_pred=7,
              config_plot=True, config_db=True):
        """
        Autoregressive integrated moving average - regressor making a
        pandemic forecast. Uses limited memory BFGS optimization (lbfgs).
        As a parameter you can provide a prognostic rate (basically uses
        new cases and of deaths) and the number of days until the forecast
        is made. Used decorators immediately save the result to the
        database and draw a forecast graph and original data.

        Parameters:
        -----------
        - keys (list) - pointers to research
        - days_pred (int) - number of days to look in the future

        Returns:
        --------
        - kwargs (dict) - dictionary for each key, icontaining data for
        drawing graphs for: orignal, train, test and forecast data.
        """
        pred_dict = {}
        kwargs = {}
        for key in keys:
            data = self.get_data_from_self()[key]

            act = len(data)
            horizont = pd.DataFrame(np.zeros(days_pred))
            ndata = pd.concat([data, horizont], ignore_index=True)

            arima_model = auto_arima(data[:act], method='lbfgs')
            test = ndata[act:]

            forecast = pd.DataFrame(arima_model.predict(n_periods=len(test)),
                                      index=test.index).values.flatten()

            x_data = list(range(1, len(data) + 1))
            x_forecast = list(range(len(data) + 1, len(data) + days_pred + 1))

            kwargs[key] = __class__.make_cap(x_data, data,
                                             [0], [0],
                                             [0], [0],
                                             x_forecast, forecast)

        return kwargs

    @plot_decorator
    @db_decorator
    def LSTM(self, keys=['New cases', 'New deaths'], days_pred=7,
             config_plot=True, config_db=False):
        """
        Long short-term memory - artificial recurrent neural network.
        Split the data in a 0.75-0.25 (train-test). To learn neural
        network, used a shift horizont of 1 day in back
        (makes the best results). Network consists of 50 units of LSTM, a
        fully connected layers: 1 and selected optimizer is 'nadam'.
        Forecast: last downloaded data is given to network
        (output of network is append to prediction_list, where in second
        iteration last data will be used to feed network).

        Parameters:
        -----------
        - keys (list) - pointers to research
        - days_pred (int) - number of days to look in the future

        Returns:
        --------
        - kwargs (dict) - dictionary for each key, icontaining data for
        drawing graphs for: orignal, train, test and forecast data.
        """
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
            train_size = int(len(dataset) * 0.65)
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
            model.add(LSTM(25, input_shape=(1, look_back)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='nadam')
            model.fit(trainX, trainY, epochs=500, batch_size=64, verbose=1)

            # Use fitted network to predict a train and test sets
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)

            # Get original values
            trainPredict = scaler.inverse_transform(trainPredict)
            testPredict = scaler.inverse_transform(testPredict)

            # Forecast the next num_pred days
            prediction_list = dataset[-look_back:]
            for _ in range(days_pred):
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
                                    len(dataset) + days_pred + 1))
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

    @plot_decorator
    @db_decorator
    def HVES(self, keys=['New cases', 'New deaths'], days_pred=7,
             config_plot=True, config_db=False):
        """
        Forecasting method using HoltWinters method. If the days_pred is
        greater, the forecast is worse.
        Parameters:
        -----------
        - keys (list) - pointers to research
        - days_pred (int) - number of days to look in the future

        Returns:
        --------
        - kwargs (dict) - dictionary for each key, icontaining data for
        drawing graphs for: orignal, train, test and forecast data.
        """
        kwargs = {}
        for key in keys:
            data = self.get_data_from_self()[key]
            data = data[10:]

            x_train = data.iloc[:-days_pred]
            model = HWES(x_train, seasonal_periods=days_pred, trend='mul',
                         seasonal='mul', initialization_method='heuristic')

            fitted = model.fit()
            forecast = fitted.forecast(steps=days_pred)
            forecast = np.array(forecast, int)

            x_data = list(range(1, len(data) + 1))
            x_forecast = list(range(len(data) + 1, len(data) + days_pred + 1))

            kwargs[key] = __class__.make_cap(x_data, data,
                                             [0], [0],
                                             [0], [0],
                                             x_forecast, forecast)
        return kwargs

    @plot_decorator
    @db_decorator
    def SARIMA(self, keys=['New cases', 'New deaths'], days_pred=7,
               config_plot=True, config_db=False):
        kwargs = {}
        for key in keys:
            data = self.get_data_from_self()[key]
            train = 0.9
            split = round(len(data)*train)
            training, testing = data[:split], data[split:]

            model = SARIMAX(training, order=(1, 1, 1),
                            seasonal_order=(2, 1, 1, 10),
                            enforce_stationarity=True,
                            enforce_invertibility=True,
                            mle_regression=False)
            model_fit = model.fit(disp=False)
            K = len(testing)
            forecast = model_fit.forecast(K)
            print(forecast)

            plt.plot(forecast, 'red')
            plt.plot(data, 'b')
            plt.axvline(x=data.index[split], color='black')
            plt.show()

        return kwargs

    @plot_decorator
    @db_decorator
    def VAR(self, keys=['New cases', 'New deaths'], days_pred=7,
             config_plot=True, config_db=False):
        """ Forecasting method using vector autoregression. """
        from statsmodels.tsa.vector_ar.var_model import VAR
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

        yhat = model_fit.forecast(model_fit.y, steps=days_pred)
        print('yhat: ', yhat)
        dict = {keys[0]: int(yhat[0][0]), keys[1]: int(yhat[0][1])}
        return dict


P = Process()
#keys = ['New cases', 'New recovered', 'New deaths']
keys1 = ['New cases']
num_pred1 = 5



#print(P.get_new_recovered())
# #P.SARIMA(keys, num_pred)
# P.LSTM(keys, num_pred)
#P.HVES(keys=keys1, days_pred=num_pred1, config_plot=True, config_db=False)
#P.ARIMA(keys, num_pred, True, False)


# P.PRED()
# cap = {'date': '01-01-01','cases_pred': 10,
#     'deaths_pred': 20}
# PredBase.insert(**cap)
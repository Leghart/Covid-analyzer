import numpy as np
import math
import pandas as pd
import textwrap
import os
import ssl
import smtplib
import datetime
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import Rbf
from sklearn.metrics import r2_score

from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest

from data_base import init_db
from data_base import MainBase
from data_base import PredBase

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")



def format_number(string):
    return " ".join(digit for digit in textwrap.wrap(
                                str(string)[::-1], 3))[::-1]

class Process:
    path = os.path.dirname(os.path.abspath(__file__))
    fields = ['new_cases', 'total_cases', 'total_recovered', 'active_cases',
                'new_deaths', 'total_deaths', 'tot_1M', 'fatality_ratio',
                'total_tests', 'date']

    def __init__(self):
        raw_data = MainBase.get_data()

        for key in __class__.fields:
            self.__dict__[key] = []

        for i in raw_data:
            for key, val in i.items():
                self.__dict__[key].append(val)

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
                p_min = [np.linalg.norm(W_rep[j] - X_std[i]) for j in range(klasy)]
                min_idx = np.where(p_min == np.amin(p_min))
                W_mod.append(min_idx[0][0])
            W_mod = np.array(W_mod)

            for i in range(len(X_std)):
                W_rep[W_mod[i]] = ((W_rep[W_mod[i]] + alfa_k * (
                                        X_std[i] - W_rep[W_mod[i]])) /
                                        np.linalg.norm(W_rep[W_mod[i]]))
            alfa_k = alfa * math.exp(-1.5 * iter_k)

        return W_rep

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

    def RBF_prediction(self, keys):
        data = self.get_data_from_self()
        data = data[:-1]

        pred_dict = {}
        for i in keys:
            plt.figure(i)
            X, y = Process.preprocessData(data, i, int(len(self.date)*0.2))
            print('Calculating prediction for {}'.format(i))
            y_rad = self.RBF(X, y, 200, 10)

            pred_dict[i] = int(y_rad[-1])

            t1 = range(len(y_rad))
            t2 = range(len(y))

            plt.plot(t1, y_rad, label="Prediction")
            plt.plot(t2, y, label="Original data")
            plt.xlabel("Days since the start of the pandemic")
            plt.legend()
            plt.title('{} ({})'.format(i,self.date[-1]))
            plt.grid()

            full_path = Process.path + '\static\{}.png'.format(i)
            if os.path.isfile(full_path):
                os.remove(full_path)

            plt.savefig(Process.path + '\static/{}'.format(i))
        self.cases_pred = pred_dict['New cases']
        self.deaths_pred = pred_dict['New deaths']

        self.next_day = (datetime.datetime.strptime(self.date[-1],'%d.%m.%Y') +
                    datetime.timedelta(days=1)).strftime('%d.%m.%Y')

    def plot_decorator(f):
        def func(self, *args):
            kwargs = f(self, *args)
            for key in kwargs:
                plt.figure(key)
                plt.plot(kwargs[key][0], label="Original data")
                plt.plot(kwargs[key][1], label="Forecast")
                plt.legend()
                plt.title('{} ({})'.format(key, self.date[-1]))
                plt.grid()

                full_path = __class__.path + '\static\{}.png'.format(key)
                if os.path.isfile(full_path):
                    os.remove(full_path)

                plt.savefig(__class__.path + '\static/{}'.format(key))

        return func

    @plot_decorator
    def ARIMA(self, keys, days_pred=7):
        pred_dict = {}
        kwargs = {}
        for key in keys:
            data = self.get_data_from_self()[key]
            act = len(data)
            horizont = pd.DataFrame(np.zeros(days_pred))
            ndata = pd.concat([data, horizont], ignore_index=True)

            print('Calculating prediction for {}'.format(key))
            arima_model =  auto_arima(data[:act], method='lbfgs')
            test = ndata[act:]

            prediction = pd.DataFrame(arima_model.predict(n_periods = len(test)), index=test.index)
            print(prediction)
            pred_dict[key] = int(prediction.values.tolist()[0][0])
            kwargs[key] = [data, prediction]


        self.cases_pred = pred_dict['New cases']
        self.deaths_pred = pred_dict['New deaths']

        self.next_day = (datetime.datetime.strptime(self.date[-1],'%d.%m.%Y') +
                    datetime.timedelta(days=1)).strftime('%d.%m.%Y')

        init_db()
        cap = {'date': self.next_day,'cases_pred': self.cases_pred,
                'deaths_pred': self.deaths_pred}
        PredBase.insert(**cap)

        return kwargs

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
                    format_number(str(self.new_cases[-1])),
                    format_number(str(self.new_deaths[-1])),
                    format_number(str(self.total_cases[-1])),
                    format_number(str(self.total_deaths[-1])),
                    format_number(str(self.total_recovered[-1])),
                    format_number(str(self.active_cases[-1])),
                    format_number(str(self.tot_1M[-1])),
                    str(self.fatality_ratio[-1]),
                    format_number(str(self.total_cases[-1])),
                    str(self.next_day),
                    format_number(str(self.cases_pred)),
                    format_number(str(self.deaths_pred)))
        return 'Subject: {}\n\n{}'.format(subject, message.encode(
                                    'ascii', 'ignore').decode('ascii'))

    def send_mail(self, broadcaster_handler, receiver_handler,
                  password_handler):

        port = 465
        smtp_server = "smtp.gmail.com"
        try:
            broadcaster = open(broadcaster_handler).read()
            receiver = open(receiver_handler).read().split(';')
            password = open(password_handler).read()
            message = self.raport_to_mail()
            del receiver[-1]

            ssl_pol = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, port, context=ssl_pol) as serwer:
                serwer.login(broadcaster, password)
                serwer.sendmail(broadcaster, receiver, message)
            print('Mail was sent!')
        except Exception as e:
            print('Error:', e)

    def ARIMA_test(self, keys, days_pred=7):
        pred_dict = {}
        kwargs = {}
        for key in keys:
            data = self.get_data_from_self()[key]
            act = len(data)
            horizont = pd.DataFrame(np.zeros(days_pred))
            ndata = pd.concat([data, horizont], ignore_index=True)
            arima_model =  auto_arima(data[:act], method='lbfgs')
            test = ndata[act:]

            prediction = pd.DataFrame(arima_model.predict(n_periods = len(test)), index=test.index)
            #prediction.columns = ['prediction']
            pred_dict[key] = int(prediction.values.tolist()[0][0])
            kwargs[key] = [data, prediction]

        return kwargs

    def VAR(self, keys=['New cases','New deaths']):
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

        # fit model
        #print(data[-3:])
        model = VAR(data)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.forecast(model_fit.y, steps=1)
        #print('{} - predykcja: {}'.format(VAR.__name__, yhat ))
        dict ={keys[0]: int(yhat[0][0]), keys[1]: int(yhat[0][1])}
        return dict

    def HVES(self, keys):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
        di = {}
        for k in keys:
            df = self.get_data_from_self()[k]
            df = df[10:]

            tmp = 3
            df_train = df.iloc[:-tmp]
            df_test = df.iloc[-tmp:]
            model = HWES(df_train, seasonal_periods=tmp, trend='mul', seasonal='add', initialization_method='estimated')
            #fitted = model.fit(optimized=True, use_brute=True)
            fitted = model.fit()
            #print out the training summary
            #print(fitted.summary())

            #create an out of sample forcast for the next 12 steps beyond the final data point in the training data set
            sales_forecast = fitted.forecast(steps=tmp)

            predi = list(sales_forecast[:1])
            di[k] = int(predi[0])
        #print('{} - predykcja : {}'.format('HVES', di ))
        return di

        '''
        fig = plt.figure()
        fig.suptitle('Retail Sales of Used Cars in the US (1992-2020)')
        past, = plt.plot(df_train.index, df_train, 'b.-', label='Sales History')
        future, = plt.plot(df_test.index, df_test, 'r.-', label='Actual Sales')
        predicted_future, = plt.plot(df_test.index, sales_forecast, 'g.-', label='Sales Forecast')
        plt.legend(handles=[past, predicted_future])
        plt.show()
        '''

    def PRED(self, keys=['New cases', 'New deaths'], Forecast_hor=7):

        next_day = (datetime.datetime.strptime(MainBase.get_data()[-1]['date'],'%d.%m.%Y') +
                    datetime.timedelta(days=1)).strftime('%d.%m.%Y')

        print('Predykcja na {}'.format(next_day) )
        #v = self.VAR(keys)
        #print('VAR: ', v)

        a = self.ARIMA_test(keys, Forecast_hor)
        print('ARIMA: ', a)

        h = self.HVES(keys)
        print('HVES: ', h)

        waga = 3
        cases = int(( a['New cases'] + h['New cases'])/waga)
        deaths = int(( + a['New deaths'] + h['New deaths'])/waga)

        print('TOTAL: ',cases,deaths)


        self.cases_pred = cases
        self.deaths_pred = deaths
        self.next_day = next_day


        init_db()
        cap = {'date': self.next_day,'cases_pred': self.cases_pred,
                'deaths_pred': self.deaths_pred}
        PredBase.insert(**cap)


#P = Process()
#P.ARIMA_test(['New cases','New deaths'])

# P.PRED()
# cap = {'date': '69-69-69','cases_pred': 10,
#     'deaths_pred': 20}
# PredBase.insert(**cap)

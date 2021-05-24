import numpy as np
import math
import pandas as pd
import textwrap
import os
import ssl
import smtplib
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from data_base import DataBase as DB


def format_number(string):
    return " ".join(digit for digit in textwrap.wrap(
                                str(string)[::-1], 3))[::-1]

class Process:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, DB):
        DB.cursor.execute('SELECT * FROM ' + DB.country)
        raw_data = DB.cursor.fetchall()

        self.country = DB.country
        self.dates = []
        self.total_cases = []
        self.new_cases = []
        self.total_deaths = []
        self.new_deaths = []
        self.total_rec = []
        self.active_cases = []
        self.tot = []
        self.total_tests = []
        self.fatality_ratio = []

        for i in raw_data:
            self.dates.append(i[0])
            self.total_cases.append(i[2])
            self.new_cases.append(i[1])
            self.total_deaths.append(i[4])
            self.new_deaths.append(i[3])
            self.total_rec.append(i[5])
            self.active_cases.append(i[6])
            self.tot.append(i[7])
            self.fatality_ratio.append(i[8])
            self.total_tests.append(i[9])

        self.cases_pred = 0
        self.deaths_pred = 0

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

    def get_data(self):
        new_data = np.arange(len(self.dates))
        d = {'Date': new_data, 'Total cases': self.total_cases,
             'New cases': self.new_cases, 'Total deaths': self.total_deaths,
             'New deaths': self.new_deaths, 'Total recovered': self.total_rec,
             'Active cases': self.active_cases, 'Tot /1M': self.tot,
             'Fatality ratio': self.fatality_ratio,
             'Total tests': self.total_tests}
        df = pd.DataFrame(data=d)
        df = df.drop(columns=['Tot /1M', 'Total tests'])
        return df

    def mlp_regress(self, key):
        X, y = Process.preprocessData(self.get_data(), key)
        X = X.astype(np.int)
        y = y.astype(np.int)
        y = np.ravel(y)

        sc = MinMaxScaler()
        X_sc = sc.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_sc, y,
                                                            test_size=0.2,
                                                            random_state=0)

        mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh',
                           solver='sgd', learning_rate='adaptive',
                           learning_rate_init=0.0001, max_iter=1000)

        mlp.fit(X_train, y_train)
        t = np.arange(0, len(y), 1).reshape(-1, 1)

        yp = mlp.predict(X_sc)
        print(f'{i}: {yp[-1]}')

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111)
        plt.plot(t, y, 'r', label='oryginalne dane')
        plt.plot(t, yp, 'b', label='predykcja')
        plt.xlabel('Numer próbki')
        plt.ylabel(key)
        plt.grid(which='both')
        plt.legend()
        plt.show()

    def Kohonen(self, X, klasy, alfa=0.5, il_iter=100):
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

    def RBF_prediction(self, keys, *args, **kwargs):
        data = self.get_data()
        pred_dict = {}
        for i in keys:
            plt.figure(i)
            X, y = Process.preprocessData(data, i, 30)
            print('Calculating prediction for {}'.format(i))
            y_rad = self.RBF(X, y, 100, 10)

            pred_dict[i] = int(y_rad[-1])

            t1 = range(len(y_rad))
            t2 = range(len(y))

            plt.plot(t1, y_rad, label="Prediction")
            plt.plot(t2, y, label="Original data")
            plt.xlabel("Days since the start of the pandemic")
            plt.legend()
            plt.title('{} ({})'.format(i,self.dates[-1]))
            plt.grid()

            full_path = Process.path + '\static\{}.png'.format(i)
            if os.path.isfile(full_path):
                os.remove(full_path)

            plt.savefig(Process.path + '\static/{}'.format(i))

        self.cases_pred = pred_dict['New cases']
        self.deaths_pred = pred_dict['New deaths']

    def save_predicion_to_txt(self, date, file):
        text = '''Prediction for {}: new cases: {}
                           new deaths: {}\n'''.format(date,
                                        self.cases_pred,
                                        self.deaths_pred)
        with open(file, 'a', encoding='utf-8') as f:
            f.write(text)
        print(text)
        print('Successfully written to file!')

    def raport_to_mail(self):
        From = "Automatyczny Raport Wirusowy"
        subject = f'Raport z dnia: {self.dates[-1]}'
        message = """
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
        ------------ Prognoza na jutro ----------------
        Zachorowania: {}\n
        Zgony: {}
        """.format(
                    format_number(str(self.new_cases[-1])),
                    format_number(str(self.new_deaths[-1])),
                    format_number(str(self.total_cases[-1])),
                    format_number(str(self.total_deaths[-1])),
                    format_number(str(self.total_rec[-1])),
                    format_number(str(self.active_cases[-1])),
                    format_number(str(self.tot[-1])),
                    str(self.fatality_ratio[-1]),
                    format_number(str(self.total_cases[-1])),
                    format_number(str(self.cases_pred)),
                    format_number(str(self.deaths_pred)))
        return 'Subject: {}\n\n{}'.format(subject, message.encode(
                                    'ascii', 'ignore').decode('ascii'))

    def send_mail(self, path_file):
        port = 465
        try:
            file = open(path_file)
            passw = file.read().split(';')
            smtp_server = "smtp.gmail.com"
            broadcaster = passw[0]
            receiver = passw[1]
            pass_ = passw[2]

            message = self.raport_to_mail()

            ssl_pol = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, port, context=ssl_pol) as serwer:
                serwer.login(broadcaster, pass_)
                serwer.sendmail(broadcaster, receiver, message)
            print('Mail was sent!')
        except Exception as e:
            print('Error:', e)

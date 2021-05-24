from flask import Flask,render_template
import logging
import datetime
import os
import pickle
import time

from data_base import DataBase as DB
from processing import Process as P
from scrap import DailyRaport as DR

app = Flask(__name__)


@app.route("/")
def main():
    D = DB('Poland')
    last_update = D.get_last_record_date()

    try:
        pickle_file = open(os.path.dirname(os.path.abspath(__file__)) +
                                            '\Process_Object.pickle','rb')
        Pl = pickle.load(pickle_file)
        pickle_file.close()

        new_cases_pred = Pl.cases_pred
        new_deaths_pred = Pl.deaths_pred
        tomorrow_date = datetime.date.today() + datetime.timedelta(days=1)
    except FileNotFoundError:
        print('File doesnt exist')
        new_cases_pred = 'Error'
        new_deaths_pred = 'Error'
        tomorrow_date = 'Error'

    if os.path.isfile(Pl.path + '\Process_Object.pickle'):
        os.remove()

    return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)

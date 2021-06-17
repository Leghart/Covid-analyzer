from flask import Flask, render_template
from flask_table import Table, Col
import logging
import datetime
import os
import time

from data_base import PredBase, MainBase, init_db

app = Flask(__name__)

@app.route("/")
def index():

    last_update = MainBase.get_last_record(get_date=True)
    pred_data = PredBase.get_last_record()
    di = {}
    for key, val in pred_data:
        di.setdefault(key, val)

    new_cases_pred = di['cases_pred']
    new_deaths_pred = di['deaths_pred']
    tomorrow_date = di['date']


    heading = ('Date','New cases','Total cases','New deaths', 'Total deaths',
                'Total recovered', 'Active cases', 'Tot per 1M',
                'Fatality ratio', 'Total tests')

    keys = ('date','new_cases','total_cases','new_deaths', 'total_deaths',
            'total_recovered', 'active_cases', 'tot_1M','fatality_ratio',
            'total_tests')

    raw_data = MainBase.get_data()[-14:]
    data = []

    for cell in raw_data:
        tmp = [cell[key] for key in keys]
        data.append(tmp)

    return render_template('main.html', **locals())



if __name__ == '__main__':
    app.run(debug=True)

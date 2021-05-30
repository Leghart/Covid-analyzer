from flask import Flask, render_template
import logging
import datetime
import os
import time

from data_base import db_session, get_last_record, insert, delete, get_data
from data_base import PredBase, MainBase, init_db

app = Flask(__name__)


@app.route("/")
def index():
    last_update = get_last_record(MainBase, get_date = True)


    pred_data = get_last_record(PredBase)
    di = {}
    for key, val in pred_data:
        di.setdefault(key, val)

    new_cases_pred = di['cases_pred']
    new_deaths_pred = di['deaths_pred']
    tomorrow_date = di['date']

    return render_template('index.html', **locals())



if __name__ == '__main__':
    app.run(debug=True)

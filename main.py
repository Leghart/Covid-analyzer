import win32gui, win32con
from flask import Flask,render_template
import logging
import datetime
import threading
import os
import time

from data_base import DataBase as DB
from processing import Process as P
from scrap import DailyRaport as DR


app = Flask(__name__)


@app.route("/")
def main():
    D = DB('Poland')
    Pl = P(D)
    last_update = D.get_last_record_date()

    # TUTAJ SA PROBLEMY BO NIE MA JESZCZE PREDYKCJI
    new_cases_pred = Pl.pred_dict['New cases']
    new_deaths_pred = Pl.pred_dict['New deaths']

    tomorrow_date = datetime.date.today() + datetime.timedelta(days=1)
    return render_template('index.html', **locals())



def show_term():
    win32gui.ShowWindow(the_program_to_hide, win32con.SW_RESTORE)

def hide_term():
    win32gui.ShowWindow(the_program_to_hide , win32con.SW_HIDE)


def collect_data():
    scrap_time = '11:15'
    update_time = '18:30'
    today = datetime.datetime.today()
    today_ = today.strftime('%d.%m.%Y')

    keys = ['New cases','New deaths']
    W = DB('Poland')
    last_day_db = W.get_last_record_date()
    message = 'Wait for a next scrap...'

    while(True):
        print(message)
        hour_now = datetime.datetime.now().hour
        min_now = datetime.datetime.now().minute
        time_now = str(hour_now) + ':' + str(min_now)
        try:
            if time_now >= scrap_time and W.get_last_record_date() != today_:
                os.system('cls')
                D = DR('Poland')
                D.show_raport()
                W.insert(D)
                Pl = P(W)
                Pl.save_plot_prediction(keys)
                Pl.send_mail(Pl.path + '\passwords')
                tomorrow_date = datetime.date.today() + (
                                datetime.timedelta(days=1))
                Pl.save_predicion_to_txt(tomorrow_date,
                                        Pl.path + '\predykcje.txt')
                message = 'Data was already downloaded.'
            else:
                message = 'Data is in a database.'
            return 0
            '''
            if time_now >= update_time:
                D = DR('Poland')
                if W.get_last_record_date() == D.date:
                    W.update(D)
                    return 0
            '''
        except Exception as e:
            print("Error: ", e)
            pass

        time.sleep(10*60)


if __name__ == '__main__':
    #x = threading.Thread(target=collect_data)
    #x.start()
    collect_data()
    app.run(debug=True)
    #collect_data()

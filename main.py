import win32gui, win32con
from flask import Flask,render_template
import logging
import datetime
import threading
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
    _predict = Pl.predict(['New cases', 'New deaths'])
    tomorrow_cases = ((_predict[0][-1]))
    tomorrow_deaths = ((_predict[1][-1]))

    new_cases_pred = int(tomorrow_cases[0])
    new_deaths_pred = int(tomorrow_deaths[0])

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

    #hide_term()
    FIRST = True
    W = DB('Poland')

    last_day_db = W.get_last_record_date()

    while(True):
        print('Wait for a next scrap')
        hour_now = datetime.datetime.now().hour
        min_now = datetime.datetime.now().minute
        time_now = str(hour_now) + ':' + str(min_now)
        try:
            if time_now >= scrap_time and FIRST:
                D = DR('Poland')
                #show_term()
                D.show_raport()
                W.insert(D)
                Pl = P(W)
                if last_day_db != today_:
                    Pl.send_mail()

                #hide_term()
                FIRST = False
                keys = ['New cases','New deaths']
                Pl = P(W)
                Pl.plot_predict(keys)
                _predict = Pl.predict(keys)
                cases_pred= int((_predict[0][-1])[0])
                deaths_pred = int((_predict[1][-1])[0])

                tomorrow_date = datetime.date.today() + (
                                datetime.timedelta(days=1))
                plik = open('predykcje.txt', 'a', encoding='utf8')
                pred = f"""Predykcja na {tomorrow_date}: zakażeń - {cases_pred}
                         zgonów - {deaths_pred}\n"""
                print(pred)
                plik.write(pred)
                plik.close()


            if time_now >= update_time:
                #show_term()
                D = DailyRaport('Poland')
                W.update(D)
                exit(0)
        except:
            pass

        time.sleep(10*60)


if __name__ == '__main__':
    x = threading.Thread(target=collect_data)
    x.start()
    app.run(debug=True)

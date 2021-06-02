import win32gui, win32con
import datetime
import os
import time
import pickle

from processing import Process
from scrap import DailyRaport as DR

from data_base import db_session, get_last_record, insert, delete, get_data
from data_base import PredBase, MainBase, init_db


def show_term():
    win32gui.ShowWindow(the_program_to_hide, win32con.SW_RESTORE)

def hide_term():
    win32gui.ShowWindow(the_program_to_hide , win32con.SW_HIDE)


if __name__ == '__main__':
    scrap_time = '11:00'
    today_ = datetime.datetime.today().strftime('%d.%m.%Y')

    keys = ['New cases','New deaths']
    message = 'Wait for a next scrap...'

    while(True):
        print(message)
        hour_now = datetime.datetime.now().hour
        min_now = datetime.datetime.now().minute
        time_now = str(hour_now) + ':' + str(min_now)
        try:
            if time_now >= scrap_time:
                if get_last_record(MainBase, get_date = True) != today_:
                    os.system('cls')
                    D = DR('Poland')
                    kwargs = D.return_cap()
                    insert(MainBase, **kwargs)

                    Pl = Process()
                    Pl.RBF_prediction(keys)

                    Pl.send_mail(Pl.path + '\passwords')
                    message = 'Data was already downloaded. Waiting for update time.'
                else:
                    message = 'Data is in a database.'
            else:
                message = 'Data wasnt uploaded yet.'

        except Exception as e:
            print("Error: ", e)
            pass

        time.sleep(5)

import datetime
import os
import time

from processing import Process
from scrap import DailyReport as DR

from data_base import PredBase, MainBase, init_db
from setup import Country, scrap_time, Forecast_hor


if __name__ == '__main__':
    today_ = datetime.datetime.today().strftime('%d.%m.%Y')

    keys = ['New cases', 'New deaths']
    message = 'Wait for a next scrap...'

    while(True):
        print(message)
        hour_now = datetime.datetime.now().hour
        min_now = datetime.datetime.now().minute
        time_now = str(hour_now) + ':' + str(min_now)
        try:
            if time_now >= scrap_time:
                if MainBase.get_last_record(get_date=True) != today_:
                    os.system('cls')
                    print('Today pred: {}'.format(PredBase.get_last_record()))
                    D = DR(Country)
                    print(D)

                    kwargs = D.return_cap()

                    MainBase.insert(**kwargs)

                    Pl = Process()
                    Pl.ARIMA(keys, Forecast_hor)
                    #Pl.LSTM(15,keys)

                    broad_file = Pl.path + r'\broadcaster'
                    rec_file = Pl.path + r'\receiver'
                    pass_file = Pl.path + r'\password'
                    Pl.send_mail(broad_file, rec_file, pass_file)
                    exit()
            else:
                message = 'Data wasnt uploaded yet.'

        except Exception as e:
            print("Error: ", e)
            pass

        time.sleep(5)

import datetime
import os
import time
from datetime import timedelta

from data_base import MainBase, PredBase
from git_handler import git_pull, git_push
from mailer import send_mail
from processing import Process
from scrapper import CollectDataException, DailyReport
from settings import COUNTRY, DATE_FORMAT, FORECAST_HOR, MAIN_PATH, OS_CON, SCRAP_TIME


def auto_get_data(today, keys):
    while True:
        hour_now = datetime.datetime.now().hour
        min_now = datetime.datetime.now().minute
        time_now = str(hour_now) + ":" + str(min_now)
        try:
            if time_now >= SCRAP_TIME:
                if MainBase.get_last_record(get_date=True) != today:
                    os.system("cls")
                    print("Today pred: {}".format(PredBase.get_last_record()))
                    daily_report = DailyReport(COUNTRY)

                    # print(daily_report)

                    report_data = daily_report.return_cap()

                    MainBase.insert(**report_data)

                    predict_process = Process()
                    predict_process.ARIMA(
                        keys=keys,
                        days_pred=FORECAST_HOR,
                        config_plot=True,
                        config_db=True,
                    )
                    return

        except CollectDataException as e:
            print(e)

        except Exception as e:
            print("Error: ", e)

        time.sleep(5)


def manual_get_data(today, keys):
    pass


def main():
    today = datetime.datetime.today().strftime(DATE_FORMAT)
    keys = ["New cases", "New deaths", "New recovered"]

    git_pull()
    date_string = MainBase.get_last_record(get_date=True)

    day_after = datetime.datetime.strptime(date_string, DATE_FORMAT) + timedelta(days=1)

    if datetime.datetime.strftime(day_after, DATE_FORMAT) != today:
        print("Database is not up to date! .")
        exit(-1)

    auto_get_data(today, keys)

    git_push(
        "Covid_Data.db",
        message="Database update: {}".format(today),
    )

    # path = MAIN_PATH + OS_CON
    # send_mail(*[path + file for file in os.listdir(path + "mail_data")])


if __name__ == "__main__":
    main()

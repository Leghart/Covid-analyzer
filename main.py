import datetime
import os
import time
from datetime import timedelta

from data_base import MainBase, PredBase
from git_handler import git_pull, git_push
from processing import Process
from scrap import CollectDataException
from scrap import DailyReport as DR
from settings import COUNTRY, DATE_FORMAT, FORECAST_HOR, OS_CON, SCRAP_TIME


def main():
    today_ = datetime.datetime.today().strftime(DATE_FORMAT)
    keys = ["New cases", "New deaths", "New recovered"]

    git_pull()
    date_string = MainBase.get_last_record(get_date=True)

    day_after = datetime.strptime(date_string, DATE_FORMAT) + timedelta(days=1)

    # last_commit_day = datetime.datetime.fromtimestamp(master.commit.committed_date)
    # print(datetime.datetime.strftime(day_after, "%d.%m.%Y"))
    # print(datetime.datetime.strftime(last_commit_day, "%d.%m.%Y"))

    if datetime.datetime.strftime(day_after, DATE_FORMAT) != today_:
        print("Database is not up to date! Program is exiting.")
        exit(-1)

    while True:
        hour_now = datetime.datetime.now().hour
        min_now = datetime.datetime.now().minute
        time_now = str(hour_now) + ":" + str(min_now)
        try:
            if time_now >= SCRAP_TIME:
                if MainBase.get_last_record(get_date=True) != today_:
                    os.system("cls")
                    print("Today pred: {}".format(PredBase.get_last_record()))
                    D = DR(COUNTRY)

                    print(D)

                    kwargs = D.return_cap()

                    MainBase.insert(**kwargs)

                    Pl = Process()
                    Pl.ARIMA(
                        keys=keys,
                        days_pred=FORECAST_HOR,
                        config_plot=True,
                        config_db=True,
                    )
                    git_push(
                        "Covid_Data.db",
                        message="Database update: {}".format(today_),
                        update_db=True,
                    )
                    broad_file = Pl.path + OS_CON + "broadcaster"
                    rec_file = Pl.path + OS_CON + "receiver"
                    pass_file = Pl.path + OS_CON + "password"
                    Pl.send_mail(broad_file, rec_file, pass_file)
                    exit(0)

        except CollectDataException as e:
            print(e)

        except Exception as e:
            print("Error: ", e)

        time.sleep(5)


if __name__ == "__main__":
    main()

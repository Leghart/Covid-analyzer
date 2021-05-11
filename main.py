from scrap import DailyRaport
from data_base import DataBase as DB
from processing import Process as P

import datetime
import time
import os
import win32gui, win32con


the_program_to_hide = win32gui.GetForegroundWindow()


def show_term():
    win32gui.ShowWindow(the_program_to_hide, win32con.SW_RESTORE)

def hide_term():
    win32gui.ShowWindow(the_program_to_hide , win32con.SW_HIDE)




scrap_time = '11:15'
update_time = '18:30'
today = datetime.datetime.today()
today_ = today.strftime('%d.%m.%Y')

if __name__ == '__main__':
    #hide_term()
    FIRST = True
    W = DB('Poland')

    last_day_db = W.get_last_record_date()

    while(True):
        hour_now = datetime.datetime.now().hour
        min_now = datetime.datetime.now().minute
        time_now = str(hour_now) + ':' + str(min_now)

        if time_now >= scrap_time and FIRST:
            D = DailyRaport('Poland')
            #show_term()
            D.show_raport()
            W.insert(D)
            if last_day_db != today_:
                W = DB('Poland')
                Pl = P(W)
                Pl.send_mail()
            time.sleep(10)
            #hide_term()
            FIRST = False

            keys = ['New cases','New deaths']
            try:
                Pl.plot_predict(keys)
            except NameError:
                pass

        if time_now >= update_time:
            #show_term()
            D = DailyRaport('Poland')
            W.update(D)
            exit(0)

        time.sleep(10*60)





'''
def get_vacc_data(self):
    url='https://www.medonet.pl/zdrowie/zdrowie-dla-kazdego,zasieg-koronawirusa-covid-19--mapa-,artykul,43602150.html'
    page=get(url)
    bs=BeautifulSoup(page.content,'html.parser')
    data=''

    for i in bs.find_all('div',class_='inlineFrame'):
        data+=str(i.get_text())+'\n'

    data=[x for x in data.splitlines() if x]

    nvacc=data[4].replace(' ','')
    tvacc=data[1].replace(' ','')
    ntests=data[7].replace(',','.')

    self.new_vaccinated=int(nvacc.split(':')[1])
    tmp_tests=ntests.split(':')[1]
    n=tmp_tests.split(' ')
    self.new_tests=float(n[1])*1000
'''

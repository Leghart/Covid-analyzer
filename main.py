from scrap import DailyRaport
from data_base import DataBase as DB
from processing import Process as P
import time



if __name__ == '__main__':
    D = DailyRaport('Poland')
    D.show_raport()
    W = DB('Poland')
    W.insert(D)
    #W.update(D)
    #D.send_mail()

    Pl = P(W)
    keys = ['New cases','New deaths']
    Pl.plot_predict(keys)











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

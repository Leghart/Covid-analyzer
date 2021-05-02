from lib import *
from Data_Base import DB

class Daily_Raport:

    def __init__(self,country):
        self.get_actual_data(country)
        self.get_vacc_data()

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

    def get_actual_data(self,country):
        url='https://www.worldometers.info/coronavirus/#main_table'
        page=get(url)
        bs=BeautifulSoup(page.content,'html.parser')
        data=''

        table_data=bs.find('table',id='main_table_countries_today')

        data=''
        for tr in table_data.find_all('tr'):
            data+=tr.get_text()

        data=data.split('\n')
        idx=data.index(country)

        end_idx=idx+14
        data=data[idx:end_idx]

        try:
            self.country=data[0]
            self.total_cases=int(data[1].replace(',',''))
            self.new_cases=int(data[2].replace('+','').replace(',',''))
            self.total_deaths=int(data[3].replace(',',''))
            self.new_deaths=int(data[4].replace('+','').replace(',',''))
            self.total_rec=int(data[5].replace(',',''))
            self.active_cases=int(data[7].replace('+','').replace(',',''))
            self.critical=int(data[8].replace('+','').replace(',',''))
            self.total_tests=int(data[11].replace(',',''))
            self.population=int(data[13].replace(',',''))

            date_today=date.today()
            self.date=date_today.strftime("%d.%m.%Y")
        except ValueError:
            print("Data wasn't uploaded on page yet.\n")
            exit(-1)

    def show_raport(self):
        print(f'Country: {self.country}')
        print(f'Total cases: {self.total_cases}')
        print(f'New cases: {self.new_cases}')
        print(f'Total deaths: {self.total_deaths}')
        print(f'New deaths: {self.new_deaths}')
        print(f'Total recoveries: {self.total_rec}')
        print(f'Actice cases: {self.active_cases}')
        print(f'Critical: {self.critical}')
        print(f'Total tests: {self.total_tests}')
        print(f'Population: {self.population}')
        print(f'Data recived: {self.date}')

    def show_raport_pl(self):
        print(f'Państwo: {self.country}')
        print(f'Wszystkie przypadki zachorowań: {self.total_cases}')
        print(f'Dzisiejsze zachorowania: {self.new_cases}')
        print(f'Wszystkie zgony: {self.total_deaths}')
        print(f'Dzisiejsze zgony: {self.new_deaths}')
        print(f'Wszyscy wyzdrowiali: {self.total_rec}')
        print(f'Aktywne przypadki: {self.active_cases}')
        print(f'Stany krytyczne: {self.critical}')
        print(f'Wszystkie wykonane testy: {self.total_tests}')
        print(f'Populacja: {self.population}')
        print(f'Dane pochodzą z dnia: {self.date}')
        print(f'Szczepienia dziś: {self.new_vaccinated}')
        print(f'Testy dziś: {self.new_tests}')


Country='Poland'
D=Daily_Raport(Country)
D.show_raport_pl()


'''
Country='Poland'
D=Daily_Raport(Country)
D.show_raport()
W=DB(Country)
W.insert(D)
'''

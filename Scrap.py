from lib import *
from Data_Base import DB

class Daily_Raport:

    def __init__(self,country):

#        self.country=country
        self.get_actual_data(country)


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

        end_idx=idx+13
        data=data[idx:end_idx]

        self.country=data[0]
        self.total_cases=int(data[1].replace(',',''))
        self.new_cases=int(data[2].replace('+','').replace(',',''))
        self.total_deaths=int(data[3].replace(',',''))
        self.new_deaths=int(data[4].replace('+','').replace(',',''))
        self.total_rec=int(data[5].replace(',',''))
        self.active_cases=int(data[6].replace('+','').replace(',',''))
        self.critical=int(data[7].replace('+','').replace(',',''))
        self.total_tests=int(data[10].replace(',',''))
        self.population=int(data[12].replace(',',''))

        date_today=date.today()
        self.date=date_today.strftime("%d/%m/%Y")


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



D=Daily_Raport('France')
D.show_raport()
W=DB("USA")
W.insert(D)

'''
D=Daily_Raport()
D.show_raport()

W=DB()
W.insert(D)
'''

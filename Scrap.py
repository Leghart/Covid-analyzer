from lib import *
from Data_Base import DB



def format_number(string):
    return  " ".join(digit for digit in textwrap.wrap(str(string)[::-1], 3))[::-1]


class Daily_Raport:

    def __init__(self,country):
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
            self.tot=int(data[9].replace('+','').replace(',','')) #na 1M
            self.total_tests=int(data[11].replace(',',''))
            self.fatality_ratio=round(self.total_deaths/self.total_cases*100,2)


            date_today=date.today()
            self.date=date_today.strftime("%d.%m.%Y")

        except ValueError:
            print(f"Data wasn't uploaded on page yet ({self.country}).\n")
            exit(-1)

    def save_data_to_csv(self,ofile,nfile,country):
        nowy_plik=open(nfile,'w')
        for filename in os.listdir(ofile):
           with open(os.path.join(ofile, filename), 'r') as f: # open in readonly mode
              plik=f.read()
              linie=plik.split('\n')
              wynik=[s for s in linie if country in s]
              print(wynik)
              nowy_plik.writelines(["%s\n" % item  for item in wynik])

    def show_raport(self):
        print(f'Country: {self.country}')
        print(f'New cases: {format_number(str(self.new_cases))}')
        print(f'New deaths: {format_number(str(self.new_deaths))}')
        print(f'Total cases: {format_number(str(self.total_cases))}')
        print(f'Total deaths: {format_number(str(self.total_deaths))}')
        print(f'Total recoveries: {format_number(str(self.total_rec))}')
        print(f'Actice cases: {format_number(str(self.active_cases))}')
        print(f'Tot cases/1M: {format_number(str(self.tot))}')
        print(f'Fatality ratio: {format_number(str(self.fatality_ratio))}')
        print(f'Total tests: {format_number(str(self.total_tests))}')
        print(f'Data recived: {self.date}')

    def raport_to_mail(self):
        From="Automatyczny Raport Wirusowy"
        subject=f'Raport z dnia: {self.date}'
        message ="""Wszystkie przypadki zachorowan: {}\n
        Dzisiejsze zachorowania: {}\n
        Wszystkie zgony: {}\n
        Dzisiejsze zgony: {}\n
        Wszyscy wyzdrowiali: {}\n
        Aktywne przypadki: {}\n
        Ilość zmarłych na 1M: {}\n
        Współczynnik smiertelnosci: {}\n
        Wszystkie wykonane testy: {}
        """.format(format_number(str(self.total_cases)),format_number(str(self.new_cases)),format_number(str(self.total_deaths)),format_number(str(self.new_deaths)),format_number(str(self.total_rec)),format_number(str(self.active_cases)),format_number(str(self.tot)),format_number(str(self.fatality_ratio)),format_number(str(self.total_cases)))

        return 'Subject: {}\n\n{}'.format(subject,message.encode('ascii', 'ignore').decode('ascii'))

    def send_mail(self):
        port=465
        file=open('passwords')
        passw=file.read().split(';')
        smtp_server="smtp.gmail.com"
        nadawca=passw[0]
        odbiorca=passw[1]
        haslo=passw[2]

        message=self.raport_to_mail()

        ssl_pol=ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server,port,context=ssl_pol) as serwer:
            serwer.login(nadawca,haslo)
            serwer.sendmail(nadawca,odbiorca,message)




D=Daily_Raport('Poland')
D.show_raport()
W=DB('Poland')
W.insert(D)
D.send_mail()



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

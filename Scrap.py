from lib import *
from Data_Base import DB

class Daily_Raport:

    def __init__(self):
        self.import_data()

    def get_actual_data(self):
        url='https://www.medonet.pl/zdrowie/zdrowie-dla-kazdego,zasieg-koronawirusa-covid-19--mapa-,artykul,43602150.html'
        page=get(url)
        bs=BeautifulSoup(page.content,'html.parser')
        data=''
        logf = open("log.txt", "a")
        try:
            for i in bs.find_all('div',class_='inlineFrame'):
                data+=str(i.get_text())+'\n'
            date=bs.find('p',class_='paragraph').get_text()
        except Exception as e:
            logf.write(f'Failed to scrap page. Error num: {str(e)} \n')
        return unicodedata.normalize("NFKC",data),date


    def import_data(self):
        text,date_=self.get_actual_data()
        data=[x for x in text.splitlines() if x]

        inf=data[9].replace(' ','').replace('+','').replace(')','').replace(":",' ').replace('(',' ')
        dead=data[15].replace(' ','').replace('+','').replace(')','').replace(":",' ').replace('(',' ')
        nvacc=data[4].replace(' ','')
        tvacc=data[1].replace(' ','')
        date_=date_.replace('.','/').replace('[','').replace(']','')

        self.new_infected=int(inf.split(' ')[2])
        self.total_infected=int(inf.split(' ')[1])

        self.new_deads=int(dead.split(' ')[2])
        self.total_deads=int(dead.split(' ')[1])

        self.new_vaccinated=int(nvacc.split(':')[1])
        self.total_vaccinated=int(tvacc.split(':')[1])

        self.source_date=date_.split(' ')[1]
        date_today=date.today()
        self.actual_date=date_today.strftime("%d/%m/%Y")


    def show_raport(self):
        print(f'Daily infected: {self.new_infected}\n'
                f'Daily deads: {self.new_deads}\n'
                f'Daily vaccinated: {self.new_vaccinated}\n\n'
                f'Total infected: {self.total_infected}\n'
                f'Total deads: {self.total_deads}\n'
                f'Total vaccinated: {self.total_vaccinated}\n'
                f'Data retrieved on : {self.source_date}\n')



D=Daily_Raport()
D.show_raport()

W=DB()
W.insert(D)

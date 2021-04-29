from lib import *
from Data_Base import DB

class Daily_Raport:

    def __init__(self):
        self.import_data()

    def get_actual_data(self):
        url='https://gazetawroclawska.pl/'
        page=get(url)
        bs=BeautifulSoup(page.content,'html.parser')
        data=''
        logf = open("log.txt", "a")

        try:
            for i in bs.find('ul',class_='atomsCoronavirusRegion__generalList'):
                data+=str(i.get_text())+'\n'
            date= bs.find('span',class_='componentsSpecialCoronavirusNumbers__updatedInfo').get_text()
        except Exception as e:
            logf.write(f'Failed to scrap page. Error num: {str(e)} ({self.source_date})\n')

        return data,date

    def import_data(self):
        text,date_=self.get_actual_data()
        data=text.split('\n')
        del data[-1]

        # change data form
        infected=(data[0].replace(' ','').replace(':',' ').replace('(',' ').replace(')',' '))
        deads=(data[1].replace(' ','').replace(':',' ').replace('(',' ').replace(')',' '))
        vaccinated=(data[2].replace(' ','').replace(':',' ').replace('(',' ').replace(')',' '))

        # get daily and total increase
        infected_prep=infected.split(' ')
        self.new_infected=int(infected_prep[1].replace('+',''))
        self.total_infected=int(infected_prep[2])

        deads_prep=deads.split(' ')
        self.new_deads=int(deads_prep[1].replace('+',''))
        self.total_deads=int(deads_prep[2])

        vacc_prep=vaccinated.split(' ')
        self.new_vaccinated=int(vacc_prep[1].replace('+',''))
        self.total_vaccinated=int(vacc_prep[2])

        date_today=date.today()
        self.actual_date=date_today.strftime("%d/%m/%Y")

        date_=date_.split(' ')
        self.source_date=date_[2].replace('.','/').replace(',','')

    def show_raport(self):
        print(f'Daily infected: {self.new_infected}\n'
                f'Daily deads: {self.new_deads}\n'
                f'Daily vaccinated: {self.new_vaccinated}\n\n'
                f'Total infected: {self.total_infected}\n'
                f'Total deads: {self.total_deads}\n'
                f'Total vaccinated: {self.total_vaccinated}\n'
                f'Data retrieved on : {self.source_date}\n')
                #f'Actual date: {self.actual_date}\n')



D=Daily_Raport()
D.import_data()
D.show_raport()

W=DB(D)

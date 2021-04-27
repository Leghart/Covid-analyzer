from bs4 import BeautifulSoup
from requests import get
from datetime import date

class Daily_Raport:
    def __init__(self):
        new_infected=0
        new_deads=0
        new_vaccinated=0

        total_infected=0
        total_deads=0
        total_vaccinated=0

        actual_date=''

    def get_actual_data(self):
        url='https://gazetawroclawska.pl/'
        page=get(url)
        bs=BeautifulSoup(page.content,'html.parser')
        data=''
        for i in bs.find('ul',class_='atomsCoronavirusRegion__generalList'):
            data+=str(i.get_text())+'\n'
        return data


    def import_data(self):
        text=self.get_actual_data()
        data=text.split('\n')
        del data[-1]

        # change data form
        infected=(data[0].replace(' ','').replace(':',' ').replace('(',' ').replace(')',' '))
        deads=(data[1].replace(' ','').replace(':',' ').replace('(',' ').replace(')',' '))
        vaccinated=(data[2].replace(' ','').replace(':',' ').replace('(',' ').replace(')',' '))


        # get daily increase
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


    def show_raport(self):
        print(f'Dzienne zakażenia: {self.new_infected}\n'
                f'Dzienne zgony: {self.new_deads}\n'
                f'Dzienne szczepienia: {self.new_vaccinated}\n'
                f'Całkowite zakażenia: {self.total_infected}\n'
                f'Całkowanie zgony: {self.total_deads}\n'
                f'Całkowite szczepienia: {self.total_vaccinated}\n')





D=Daily_Raport()
D.import_data()
D.show_raport()
print(D.actual_date)
#print(date.today())















#chrome_options = Options()
#chrome_options.add_argument('--headless')
#driver = webdriver.Chrome(executable_path=r'C:\TestFiles\chromedriver.exe',chrome_options=chrome_options)
#driver=webdriver.PhantomJS(executable_path=r'C:\TestFiles\chromedriver.exe')


#a=driver.find_elements_by_css_selector("calcite ember-application")


#path = """/html/body/div/div/div/div/div/div/margin-container/full-container/div[3]"""
#search_input = driver.find_element_by_xpath(path)
#print(search_input.get_attribute('placeholder'))





#html = driver.page_source

#cos=driver.find_elements_by_class_name('external-html')
#print(cos)

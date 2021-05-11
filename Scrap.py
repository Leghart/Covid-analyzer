from bs4 import BeautifulSoup
from requests import get
from datetime import date
import textwrap
from data_base import DataBase as DB


def format_number(string):
    return " ".join(digit for digit in textwrap.wrap(
                                str(string)[::-1], 3))[::-1]


class DailyRaport:

    def __init__(self, country):
        self.get_actual_data(country)

    def get_actual_data(self, country):
        url = 'https://www.worldometers.info/coronavirus/#main_table'
        page = get(url)
        bs = BeautifulSoup(page.content, 'html.parser')

        table_data = bs.find('table', id='main_table_countries_today')

        data = ''
        for tr in table_data.find_all('tr'):
            data += tr.get_text()

        data = data.split('\n')
        idx = data.index(country)

        end_idx = idx + 14
        data = data[idx:end_idx]

        try:
            self.country = data[0]
            self.total_cases = int(data[1].replace(',', ''))
            self.new_cases = int(data[2].replace('+', '').replace(',', ''))
            self.total_deaths = int(data[3].replace(',', ''))
            self.new_deaths = int(data[4].replace('+', '').replace(',', ''))
            self.total_rec = int(data[5].replace(',', ''))
            self.active_cases = int(data[7].replace('+', '').replace(',', ''))
            self.tot = int(data[9].replace('+', '').replace(',', ''))
            self.total_tests = int(data[11].replace(',', ''))
            self.fatality_ratio = round(self.total_deaths /
                                        self.total_cases * 100, 2)

            date_today = date.today()
            self.date = date_today.strftime("%d.%m.%Y")

        except ValueError:
            print(f"Data wasn't uploaded on page yet ({self.country}).\n")
            exit(-1)

    def save_data_to_csv(self, ofile, nfile, country):
        out_file = open(nfile, 'w')
        for filename in os.listdir(ofile):
            with open(os.path.join(ofile, filename), 'r') as f:
                day = f.read()
                lines = day.split('\n')
                out = [s for s in lines if country in s]
                out_file.writelines(["%s\n" % item for item in out])

    def show_raport(self):
        print(f'Country: {self.country}')
        print(f'New cases: {format_number(str(self.new_cases))}')
        print(f'New deaths: {format_number(str(self.new_deaths))}')
        print(f'Total cases: {format_number(str(self.total_cases))}')
        print(f'Total deaths: {format_number(str(self.total_deaths))}')
        print(f'Total recoveries: {format_number(str(self.total_rec))}')
        print(f'Actice cases: {format_number(str(self.active_cases))}')
        print(f'Tot cases/1M: {format_number(str(self.tot))}')
        print(f'Fatality ratio: {str(self.fatality_ratio)}')
        print(f'Total tests: {format_number(str(self.total_tests))}')
        print(f'Data recived: {self.date}')

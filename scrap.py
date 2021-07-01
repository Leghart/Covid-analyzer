from bs4 import BeautifulSoup
from requests import get
from datetime import date
import os
import textwrap


class DailyReport:
    """
    Class which is responsible for connect with www page where every day
    are published data about daily coronavirus cases and scrape data to get
    nessesery inforamtions.
    """

    def __init__(self, country='Poland'):
        """
        Constructor downloaded data.

        Parameters:
        -----------
        - country (string) - country from which data will be download.
         Please check if your country exists in www.worldometers.com

        Returns:
        --------
        - None
        """
        self.get_actual_data(country)

    def __str__(self):
        """
        Simple report in terminal, showing daily data.

        Parameters:
        -----------
        - None

        Returns:
        --------
        - (string) - prepared message for printing, informing about
        the daily report
        """
        return('Country: {}\nNew cases: {}\nNew deaths: {}\nTotal cases: {}\n'
               'Total deaths: {}\nTotal recovered: {}\nActive cases: {}\n'
               'Tot cases/1M: {}\nFatality ratio: {}\nTotal tests: {}\n'
               'Data recived: {}'.format(
                self.country,
                __class__.format_number(str(self.new_cases)),
                __class__.format_number(str(self.new_deaths)),
                __class__.format_number(str(self.total_cases)),
                __class__.format_number(str(self.total_deaths)),
                __class__.format_number(str(self.total_recovered)),
                __class__.format_number(str(self.active_cases)),
                __class__.format_number(str(self.tot_1M)),
                str(self.fatality_ratio),
                __class__.format_number(str(self.total_tests)),
                self.date))

    @staticmethod
    def format_number(string):
        """
        Method to easly-read number format
        (e.g. 7 521 642 instead of 7521642).

        Parameters:
        -----------
        - string (string) - number to change format

        Returns:
        --------
        - (string) - changed string
        """
        return " ".join(digit for digit in textwrap.wrap(
                                    str(string)[::-1], 3))[::-1]

    def get_actual_data(self, country='Poland'):
        """
        Main feature that connects to www.worldometers.com to download
        daily report from the selected country. An error may appear while
        retrieving data because when the function is called, the data may not
        yet be loaded into the page (will this is signaled by a special message
        in the terminal). Downloaded data is saving in existing instance
        storing main inforamtions.

        Parameters:
        -----------
        - country (string) - country from which data will be download.
         Please check if your country exists in www.worldometers.com

        Returns:
        --------
        - None
        """
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
            self.new_cases = int(data[2].replace('+', '').replace(',', ''))
        except ValueError:
            self.new_cases = 0

        try:
            self.new_deaths = int(data[4].replace('+', '').replace(',', ''))
        except ValueError:
            self.new_deaths = 0

        finally:
            self.country = data[0]
            self.total_cases = int(data[1].replace(',', ''))
            self.total_deaths = int(data[3].replace(',', ''))
            self.total_recovered = int(data[5].replace(',', ''))
            self.active_cases = int(data[7].replace('+', '').replace(',', ''))
            self.tot_1M = int(data[9].replace('+', '').replace(',', ''))
            self.total_tests = int(data[11].replace(',', ''))
            self.fatality_ratio = round(self.total_deaths /
                                        self.total_cases * 100, 2)
            self.date = date.today().strftime("%d.%m.%Y")


    def return_cap(self):
        """
        Return capsule with data to use it as kwargs during insert
        to database.

        Parameters:
        -----------
        - None

        Returns:
        --------
        - None
        """
        return {'new_cases': self.new_cases, 'total_cases': self.total_cases,
                'total_recovered': self.total_recovered,
                'active_cases': self.active_cases,
                'new_deaths': self.new_deaths,
                'total_deaths': self.total_deaths, 'tot_1M': self.tot_1M,
                'fatality_ratio': self.fatality_ratio,
                'total_tests': self.total_tests, 'date': self.date}

    def save_data_to_csv(self, ofile, nfile, country):
        """
        Saves the data of one country in a csv file. Data taken from:
        https://github.com/CSSEGISandData/COVID-19

        Parameters:
        -----------
        - ofile (string) - path to folder where the daily data files
        are located
        - nfile (string) - path to file where you want to save data for the
        selected country
        - country (string) - country name

        Returns:
        --------
        - None
        """
        out_file = open(nfile, 'w')
        for filename in os.listdir(ofile):
            with open(os.path.join(ofile, filename), 'r') as f:
                day = f.read()
                lines = day.split('\n')
                out = [s for s in lines if country in s]
                out_file.writelines(["%s\n" % item for item in out])

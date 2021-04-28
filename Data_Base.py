#from prog import Daily_Raport
import Scrap as S #Data_Object
from lib import *

class DB:

    def __init__(self,S):
        self.con=None
        self.cursor=None

    def open(self):
        self.con=sqlite3.connect('Covid_Data.db')
        self.cursor=self.con.cursor()

        # check existing table
        self.cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='data' ''')

        logf = open("log.txt", "a")
        if self.cursor.fetchone()[0]!=1:
            self.cursor.execute('''CREATE TABLE data (date DATETIME, new_infected INT, new_deads INT, new_vaccinated INT, total_infected INT, total_deads INT, total_vaccinated INT )''')
            logf.write('Table not existing. Created new table.\n')

    def commit(self):
        logf=open('log.txt','a')
        try:
            self.con.commit()
            self.cursor.close()
            self.con.close()
            logf.write(f"Data was successfully committed ({S.source_date}).\n")
        except Exception:
            logf.write("Data coudn't be commit.\n")


    def get_last_record_date(self):
        self.open()
        self.cursor.execute("SELECT * FROM data ORDER BY date DESC LIMIT 1")
        result=self.cursor.fetchone()
        return result[0]


    def insert(self,S):
        self.open()
        logf=open('log.txt','a')
        file=open('raports.txt','a')
        if self.get_last_record_date()!=S.source_date:
            self.cursor.execute('INSERT INTO data VALUES (?,?,?,?,?,?,?)',(S.actual_date,S.new_infected,S.new_deads,S.new_vaccinated,S.total_infected,S.total_deads,S.total_vaccinated))
            self.commit()
            file.write(str(self.actual_date)+' '+str(self.new_infected)+' '+str(self.new_deads)+' '+str(self.new_vaccinated)+' '+str(self.total_infected)+' '+str(self.total_deads)+' '+str(self.total_vaccinated)+'\n')

        else:
            logf.write(f"That record is already in data base ({S.source_date}).\n")

    #def check_last_record()

#from prog import Daily_Raport
#import Scrap as S #Data_Object
from lib import *

class DB:

    def __init__(self):
        self.con=sqlite3.connect('Covid_Data.db')
        self.cursor=self.con.cursor()

        # check existing table
        self.cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='data' ''')

        if self.cursor.fetchone()[0]!=1:
            self.cursor.execute('''CREATE TABLE data (date DATETIME, new_infected INT, new_deads INT, new_vaccinated INT, total_infected INT, total_deads INT, total_vaccinated INT )''')
            logf = open("log.txt", "a")
            logf.write('Table not existing. Created new table.\n')
            logf.close()
            

    def commit(self,S):
        logf=open('log.txt','a')
        try:
            self.con.commit()
            self.cursor.close()
            self.con.close()
            logf.write(f"Data was successfully committed ({S.source_date}).\n")
            logf.close()
        except Exception:
            logf.write("Data coudn't be commit.\n")
            logf.close()


    def get_last_record_date(self):
        self.cursor.execute('''SELECT date FROM data ''')
        return self.cursor.fetchall()[-1][0]


    def insert(self,S):
        logf=open('log.txt','a')
        file=open('raports.txt','a')
        if self.get_last_record_date()!=S.source_date:
            print(self.get_last_record_date())
            print(S.source_date)
            self.cursor.execute('INSERT INTO data VALUES (?,?,?,?,?,?,?)',(S.actual_date,S.new_infected,S.new_deads,S.new_vaccinated,S.total_infected,S.total_deads,S.total_vaccinated))
            file.write(str(S.actual_date)+' '+str(S.new_infected)+' '+str(S.new_deads)+' '+str(S.new_vaccinated)+' '+str(S.total_infected)+' '+str(S.total_deads)+' '+str(S.total_vaccinated)+'\n')
            self.commit(S)
        else:
            logf.write(f"That record is already in data base ({S.source_date}).\n")

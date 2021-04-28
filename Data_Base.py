#from prog import Daily_Raport
import prog as DO #Data_Object
from lib import *

class DB:

    def __init__(self,DO):
        self.con=None
        self.cursor=None

    def open(self):
        self.con=sqlite3.connect('Covid_Data.db')
        self.cursor=self.con.cursor()

        # check existing table
        self.cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='data' ''')

        logf = open("log.txt", "w")
        if self.cursor.fetchone()[0]!=1:
            self.cursor.execute('''CREATE TABLE data (date DATETIME, new_infected INT, new_deads INT, new_vaccinated INT, total_infected INT, total_deads INT, total_vaccinated INT )''')
            logf.write('Table not existing. Created new table.\n')

    def commit(self):
        logf=open('log.txt','w')
        try:
            self.con.commit()
            self.cursor.close()
            self.con.close()
            logf.write("Data was successfully committed.\n")
        except Exception:
            logf.write("Data coudn't be commit.\n")


    def get_last_record(self):
        self.open()
        self.cursor.execute("SELECT * FROM data ORDER BY date DESC LIMIT 1")
        result=self.cursor.fetchone()
        return result[0]

    def insert(self,DO):
        self.open()
        if self.get_last_record()!=DO.source_date: #????
            print("push")
            self.cursor.execute('INSERT INTO data VALUES (?,?,?,?,?,?,?)',(DO.actual_date,DO.new_infected,DO.new_deads,DO.new_vaccinated,DO.total_infected,DO.total_deads,DO.total_vaccinated))
            self.commit()
        else:
            print("ten record juz byl")

    #def check_last_record()

#from prog import Daily_Raport
import prog as Data_Object
from lib import *

class DB:
    def __init__(self,Data_Object):
        self.con=None
        self.cursor=None
        #print(Data_Object.new_deads)
        #con=sql.connect('Covid_Data.db')
        #cur = con.cursor()
    def open(self):
        #try:
        self.con=sqlite3.connect('Covid_Data.db')
        self.cursor=self.con.cursor()

        #self.cursor.execute('''CREATE TABLE data (date DATETIME, new_infected INT, new_deads INT, new_vaccinated, total_infected INT, total_deads INT, total_vaccinated INT )''')
        self.cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='data' ''')
        if self.cursor.fetchone()[0]==1:
            print('tabela istnieje')
        else:
            self.cursor.execute('''CREATE TABLE data (date DATETIME, new_infected INT, new_deads INT, new_vaccinated, total_infected INT, total_deads INT, total_vaccinated INT )''')
        #except sqlite3.Error:
            #print("Cant connect to database!")

    def commit_close(self):
        self.con.commit()
        self.cursor.close()
        self.con.close()

    def insert(self,Data_Object):
        self.cursor.execute('INSERT INTO data VALUES (?,?,?,?,?,?,?)',(Data_Object.actual_date,Data_Object.new_infected,Data_Object.new_deads,Data_Object.new_vaccinated,Data_Object.total_infected,Data_Object.total_deads,Data_Object.total_vaccinated))

    #def dodaj(self):
        # Create table
        #cur.execute('''CREATE TABLE data (date DATETIME, new_infected INT, new_deads INT, new_vaccinated, total_infected INT, total_deads INT, total_vaccinated INT )''')

        # Insert a row of data
        #cur.execute("INSERT INTO data VALUES ('kupa','10')")

        #con.commit()
        #con.close()

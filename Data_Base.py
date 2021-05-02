from lib import *

class DB:

    def __init__(self,country):
        self.con=sqlite3.connect('Covid_Data.db')
        self.cursor=self.con.cursor()
        self.country=country
        # check existing table
        self.cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='"+self.country+"' ")

        if self.cursor.fetchone()[0]!=1:
            self.cursor.execute("CREATE TABLE "+self.country+"(date DATETIME, new_infected INT, total_infected INT, new_deads INT, total_deads INT, total_recovered INT, active_cases INT, critical_cases INT, total_tests INT, population INT )")
            logf = open("log.txt", "a")
            logf.write('Table not existing. Created new table.\n')
            logf.close()

    def commit(self,S):
        logf=open('log.txt','a')
        try:
            self.con.commit()
            self.cursor.close()
            self.con.close()
            logf.write(f"Data was successfully committed ({S.date}).\n")
            logf.close()
        except Exception:
            logf.write("Data coudn't be commit.\n")
            logf.close()

    def get_last_record_date(self):
        self.cursor.execute("SELECT date FROM "+self.country )
        try:
            last_date=self.cursor.fetchall()[-1][0]
        except Exception:
            last_date=None
        return last_date

    def insert(self,S):
        logf=open('log.txt','a')
        file=open('raports.txt','a')
        if self.get_last_record_date()!=S.date:
            self.cursor.execute("INSERT INTO "+self.country+ " VALUES (?,?,?,?,?,?,?,?,?,?)",(S.date, S.new_cases, S.total_cases, S.new_deaths, S.total_deaths, S.total_rec, S.active_cases, S.critical, S.total_tests, S.population))
            file.write(str(self.country)+ ' ' +str(S.date)+' '+str(S.new_cases)+' '+str(S.total_cases)+' '+str(S.new_deaths)+' '+str(S.total_deaths)+' '+str(S.total_rec)+ ' '+str(S.total_rec)+' '+str(S.active_cases)+' '+str(S.critical)+' '+str(S.total_tests)+' '+str(S.population)+'\n')
            self.commit(S)
        else:
            logf.write(f"That record is already in data base ({S.date}).\n")

    def get_old_data(self):
        file='Zakazenia30323112020.csv'
        file=open(file)#,encoding="utf-8")
        data=file.read()

        df=pd.DataFrame([x.split(';') for x in data.split('\n')])
        df=df.set_axis(['Dzien','Data','Nowe przypadki','Wszystkie przypadki','Zgony','Wszystkie zgony','Ozdrowieńcy (dzienna)','Wyzdrowiali','Aktywne przypadki','Kwarantanna','Nadzór'],axis='columns')
        df=df.drop(columns=['Dzien','Ozdrowieńcy (dzienna)','Kwarantanna','Nadzór'])

        return df




D=DB('Poland')
D.get_old_data()

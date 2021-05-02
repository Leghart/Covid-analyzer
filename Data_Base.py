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
            logf.write(f'Table not existing. Created table: {self.country}.\n')
            logf.close()
        else:
            print("successful database connection.\n")

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


    def get_actual_data(self):
        self.cursor.execute("SELECT * FROM "+self.country )
        data=self.cursor.fetchall()

        df=pd.DataFrame(list(data))
        df=df.set_axis(['Data','Nowe przypadki','Wszystkie przypadki','Zgony','Wszystkie zgony,','Wyzdrowiali','Aktywne przypadki','Critical_cases','Total_tests','Population'],axis='columns')
        df=df.drop(columns=['Data','Critical_cases','Total_tests','Population'])
        #df=df.drop([0,1,2,3])
        return df


    def insert(self,S):
        logf=open('log.txt','a')
        file=open('raports.txt','a')
        if self.get_last_record_date()!=S.date:
            self.cursor.execute("INSERT INTO "+self.country+ " VALUES (?,?,?,?,?,?,?,?,?,?)",(S.date, S.new_cases, S.total_cases, S.new_deaths, S.total_deaths, S.total_rec, S.active_cases, S.critical, S.total_tests, S.population))
            file.write(str(self.country)+ ' ' +str(S.date)+' '+str(S.new_cases)+' '+str(S.total_cases)+' '+str(S.new_deaths)+' '+str(S.total_deaths)+' '+str(S.total_rec)+ ' '+str(S.total_rec)+' '+str(S.active_cases)+' '+str(S.critical)+' '+str(S.total_tests)+' '+str(S.population)+'\n')
            self.commit(S)
        else:
            logf.write(f"That record is already in data base ({S.date}).\n")


#D=DB('Poland')

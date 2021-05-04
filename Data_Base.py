from lib import *

class DB:

    def __init__(self,country):
        self.con=sqlite3.connect('Covid_Data.db')
        self.cursor=self.con.cursor()
        self.country=country
        # check existing table
        self.cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='"+self.country+"' ")

        if self.cursor.fetchone()[0]!=1:
            self.cursor.execute("CREATE TABLE "+self.country+"(date TEXT, new_cases INT, total_cases INT, new_deaths INT, total_deaths INT, total_recovered INT, active_cases INT, tot_1M REAL, fatality_ratio REAL, total_tests INT)")
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
        df=df.set_axis(['Date','New cases','Total cases','New deaths','Total deaths,','Total recovered','Active cases','Tot /1M','Fatality ratio','Total_tests'],axis='columns')
        df=df.drop(columns=['Data'])
        return df

    def insert(self,S):
        logf=open('log.txt','a')
        file=open('raports.txt','a')
        if self.get_last_record_date()!=S.date:
            self.cursor.execute("INSERT INTO "+self.country+ " VALUES (?,?,?,?,?,?,?,?,?,?)",(S.date, S.new_cases, S.total_cases, S.new_deaths, S.total_deaths, S.total_rec, S.active_cases, S.tot, S.fatality_ratio,S.total_tests))
            file.write(str(self.country)+ ' ' +str(S.date)+' '+str(S.new_cases)+' '+str(S.total_cases)+' '+str(S.new_deaths)+' '+str(S.total_deaths)+' '+str(S.total_rec)+ ' '+str(S.active_cases)+' '+str(S.tot)+' '+str(S.fatality_ratio)+' '+str(S.total_tests)+'\n')
            self.commit(S)
        else:
            logf.write(f"That record is already in data base ({S.date}).\n")

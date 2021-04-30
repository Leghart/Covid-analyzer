from lib import *
import Data_Base as DB

class Process:
    def __init__(self,DB):
        DB.cursor.execute('SELECT * FROM data')
        raw_data=DB.cursor.fetchall()

        self.dates=[i[0] for i in raw_data]
        self.ninf=[i[1] for i in raw_data]
        self.ndead=[i[2] for i in raw_data]
        self.nvac=[i[3] for i in raw_data]
        self.tinf=[i[4] for i in raw_data]
        self.tdead=[i[5] for i in raw_data]
        self.tvac=[i[6] for i in raw_data]


    def plot_new_infected(self):
        plt.figure(figsize=(10,5))
        #plt.plot(self.dates,self.ninf,label='zakażenia')
        plt.plot(self.dates,self.ndead,label='zgony')
        print(self.ndead)
        #plt.plot(self.dates,self.nvac,label='zaszczepienia')
        plt.grid(True,which='major')
        plt.ylabel('Ilość osób')
        plt.legend()
        plt.show()

from lib import *
import Data_Base as DB

class Process:
    def __init__(self,DB):
        DB.cursor.execute('SELECT * FROM data')
        raw_data=DB.cursor.fetchall()

        for i in raw_data:
            self.dates=i[0]
            self.ninf=i[1]
            self.ndead=i[2]
            self.nvac=i[3]
            self.tinf=i[4]
            self.tdead=i[5]
            self.tvac=i[6]


    def plot_new_infected(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.dates,self.ndead,label='zgony')
        print(self.ndead)
        plt.grid(True,which='major')
        plt.ylabel('Ilość osób')
        plt.legend()
        plt.show()

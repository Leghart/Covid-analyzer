from lib import *
#import Data_Base as Db
from Data_Base import DB

class Process:

    def __init__(self,DB):
        DB.cursor.execute('SELECT * FROM '+DB.country)
        raw_data=DB.cursor.fetchall()


        self.country=DB.country
        self.dates=[]
        self.total_cases=[]
        self.new_cases=[]
        self.total_deaths=[]
        self.new_deaths=[]
        self.total_rec=[]
        self.active_cases=[]
        self.critical=[]
        self.total_tests=[]
        self.population=[]

        for i in raw_data:
            self.dates.append(i[0])
            self.total_cases.append(i[2])
            self.new_cases.append(i[1])
            self.total_deaths.append(i[4])
            self.new_deaths.append(i[3])
            self.total_rec.append(i[5])
            self.active_cases.append(i[6])
            self.critical.append(i[7])
            self.total_tests.append(i[8])
            self.population.append(i[9])


    def get_old_data(self):
        file='Zakazenia30323112020.csv'
        file=open(file)#,encoding="utf-8")
        data=file.read()

        df=pd.DataFrame([x.split(';') for x in data.split('\n')])
        df=df.set_axis(['Dzien','Data','Nowe przypadki','Wszystkie przypadki','Zgony','Wszystkie zgony','Ozdrowieńcy (dzienna)','Wyzdrowiali','Aktywne przypadki','Kwarantanna','Nadzór'],axis='columns')
        df=df.drop(columns=['Data','Dzien','Ozdrowieńcy (dzienna)','Kwarantanna','Nadzór'])
        #print(df)
        return df


    def plot_new_infected(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.dates,self.ndead,label='zgony')
        plt.grid(True,which='major')
        plt.ylabel('Ilość osób')
        plt.legend()
        plt.show()


    def preprocessData(self,data,wyjscie, k=2):
        X,Y = [],[]
        for i in range(len(data)-k-1):
            x_i_mat=np.array(data[i:(i+k)])
            x_i = x_i_mat.reshape(x_i_mat.shape[0]*x_i_mat.shape[1])
            y_i= np.array(data[(i+k):(i+k+1)][wyjscie])
            X.append(x_i)
            Y.append(y_i)
        return np.array(X),np.array(Y)


    def predix(self,DB):
        key='Zgony'
        X,y = self.preprocessData(self.get_old_data(), key)
        X=X.astype(np.int)
        y=y.astype(np.int)
        y=np.ravel(y)

        sc=MinMaxScaler()
        X_sc=sc.fit_transform(X)

        X_train,X_test,y_train,y_test=train_test_split(X_sc, y, test_size= 1, random_state=0)

        mlp = MLPRegressor( hidden_layer_sizes=(100,),  activation='tanh', solver='sgd',
        learning_rate='adaptive',learning_rate_init=0.0001, max_iter=1000)
        mlp.fit(X_train,y_train)
        t=np.arange(0,len(y),1).reshape(-1,1)


        Xn,yn = self.preprocessData(DB.get_actual_data(), key)
        Xn=Xn.astype(np.int)
        #print(Xn)
        X_sc_nowe=sc.fit_transform(Xn)

        yp=mlp.predict(X_sc_nowe)

        print(yp)
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(111)
        plt.plot(t,y,'r',label='oryginalne dane')
        #plt.plot(t,yp,'b',label='predykcja')
        plt.xlabel('Numer próbki')
        plt.ylabel('Zgony')
        plt.grid(which='both')
        plt.legend()
        plt.show()


    def gen_data(self,DB):
        plik=open('data.txt','w')

        df1=self.get_old_data()
        df2=DB.get_actual_data()
        df1.to_csv (r'data.csv', index = False, header=True)
            #print(DB.get_actual_data())
            #print(self.get_old_data())
        plik.close()



D=DB('Poland')
P=Process(D)
#P.gen_data(D)
#P.get_old_data()
#P.predix(D)

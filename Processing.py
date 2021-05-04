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
        self.tot=[]
        self.total_tests=[]
        self.fatality_ratio=[]

        for i in raw_data:
            self.dates.append(i[0])
            self.total_cases.append(i[2])
            self.new_cases.append(i[1])
            self.total_deaths.append(i[4])
            self.new_deaths.append(i[3])
            self.total_rec.append(i[5])
            self.active_cases.append(i[6])
            self.tot.append(i[7])
            self.fatality_ratio.append(i[8])
            self.total_tests.append(i[9])

    def get_data(self):
        nowa_data=np.arange(len(self.dates))
        d={'Date':nowa_data,'Total cases':self.total_cases,'New cases':self.new_cases,'Total deaths':self.total_deaths,'New deaths':self.new_deaths,'Total recovered':self.total_rec,'Active cases':self.active_cases,'Tot /1M':self.tot,'Fatality ratio':self.fatality_ratio,'Total tests':self.total_tests}
        df=pd.DataFrame(data=d)
        df=df.drop(columns=['Tot /1M','Total tests'])
        return df


    def preprocessData(self,data,wyjscie, k=5):
        X,Y = [],[]
        for i in range(len(data)-k-1):
            x_i_mat=np.array(data[i:(i+k)])
            x_i = x_i_mat.reshape(x_i_mat.shape[0]*x_i_mat.shape[1])
            y_i= np.array(data[(i+k):(i+k+1)][wyjscie])
            X.append(x_i)
            Y.append(y_i)
        return np.array(X),np.array(Y)


    def predix(self):
        key='New deaths'
        X,y = self.preprocessData(self.get_data(), key)
        X=X.astype(np.int)
        y=y.astype(np.int)
        y=np.ravel(y)

        sc=MinMaxScaler()
        X_sc=sc.fit_transform(X)

        X_train,X_test,y_train,y_test=train_test_split(X_sc, y, test_size= 0.2, random_state=0)

        mlp = MLPRegressor( hidden_layer_sizes=(100,),  activation='tanh', solver='sgd',
        learning_rate='adaptive',learning_rate_init=0.0001, max_iter=1000)
        mlp.fit(X_train,y_train)

        t=np.arange(0,len(y),1).reshape(-1,1)

        yp=mlp.predict(X_sc)
        print(yp)

        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(111)
        plt.plot(t,y,'r',label='oryginalne dane')
        plt.plot(t,yp,'b',label='predykcja')
        plt.xlabel('Numer próbki')
        plt.ylabel(key)
        plt.grid(which='both')
        plt.legend()
        plt.show()




def Normalizacja(X):
    srd = 1 / len(X) * sum(i for i in X)
    X_std = []
    for i in range(len(X)):
        X_std.append((srd - X[i]) / np.linalg.norm(X[i]))
    X_std=np.array(X_std)
    return X_std

def WybMiar(X_std, W_rep, miara, klasy):
    W_mod=[]
    if miara == 1:
        for i in range(len(X_std)):
            p_max = []
            for j in range(klasy):
                p_max.append(np.dot(W_rep[j], X_std[i]))
            max_idx = np.where(p_max == np.amax(p_max))
            W_mod.append(max_idx[0][0])
        return W_mod

    if miara == 2:
        for i in range(len(X_std)):
            p_min = []
            for j in range(klasy):
                p_min.append(np.linalg.norm(W_rep[j] - X_std[i]))
            min_idx = np.where(p_min == np.amin(p_min))
            W_mod.append(min_idx[0][0])
        return W_mod

    if miara == 3:
        for i in range(len(X_std)):
            p_min = []
            for j in range(klasy):
                sum_tmp = 0
                for k in range(len([0])):
                    sum_tmp += abs(W_rep[j][k] - X_std[i][k])
                p_min.append(math.sqrt(sum_tmp))
            min_idx = np.where(p_min == np.amin(p_min))
            W_mod.append(min_idx[0][0])
        return W_mod

def InitWekRep(X,klasy):
    gen = np.random.RandomState(100)
    W_rep = []
    for i in range(klasy):
        wtemp = gen.normal(loc=0.0, scale=0.01, size=len(X[0]))
        W_rep.append(wtemp/np.linalg.norm(wtemp))
    return W_rep


def Kohonen(X, klasy, alfa=0.5, il_iter=50, miara=2, wsp_alfa=2):
    X_std = Normalizacja(X)
    W_rep=InitWekRep(X,klasy)
    alfa_k = alfa
#=============== wybor miary =================
    for iter_k in range(il_iter):
        W_mod=WybMiar(X_std,W_rep,miara,klasy)
        W_mod_indx = np.array(W_mod)

# =========================== modyfikacja wektorów reprezentantów========================
        for i in range(len(X_std)):
            W_rep[W_mod_indx[i]] = W_rep[W_mod_indx[i]] + alfa_k * (X_std[i] - W_rep[W_mod_indx[i]])
            W_rep[W_mod_indx[i]] = (W_rep[W_mod_indx[i]]) / (np.linalg.norm(W_rep[W_mod_indx[i]]))

#====================== Wybor wspolczynnika uczenia ===============
        C1=1.5
        C2=1.7
        if wsp_alfa == 1: #liniowe
            alfa_k=alfa*(il_iter-iter_k)/il_iter
        if wsp_alfa == 2: #wykladnicze
            alfa_k = alfa * math.exp(-C1*iter_k)
        if wsp_alfa == 3: #hiperboliczne
            alfa_k = C1/(C2 + iter_k)

    return W_rep

def RBF(X,y,liczba_klas,l):
    liczba_iter=50
    Xn=Normalizacja(X)
    alfa=0.5
    miara=2
    C = Kohonen(X, liczba_klas, alfa, liczba_iter, miara, wsp_alfa=2)

    #================ najdalej oddalone od siebie wzorce w zbiorach =================
    S=[]
    for i in range(len(Xn)):
        for j in range(len(Xn)):
            S.append(np.linalg.norm(Xn[i]-Xn[j]))
    S=max(S)

    #============ promien funkcji phi ================
    r = S/l

    #===== Wyznaczanie macierzy PHI ================
    PHI=[]
    for N in range (len(Xn)):
        pom=[]
        for p in range (liczba_klas):
            odl = np.linalg.norm(Xn[N]-C[p])
            pom.append(np.exp(-np.square(odl/r)))
        PHI.append(pom)
    PHI = np.array(PHI)

    cz1= np.linalg.pinv(np.dot(np.transpose(PHI),PHI))
    cz2=np.dot(np.transpose(PHI),y)
    w=np.dot(cz1,cz2)

    #======================= Wyjscie sieci ===============================
    y_rad=[]
    for i in range(len(Xn)):
        tmp = 0
        for j in range(len(w)):
            tmp += (w[j] * np.exp(-np.square(np.linalg.norm(Xn[i] - C[j])/r)))
        y_rad.append(tmp)
    return y_rad



D=DB('Poland')
P=Process(D)

key='Total cases'
dane=P.get_data()
wyjscie=[]
wek=['Date','Total cases','New cases','Total deaths','New deaths','Total recovered','Active cases','Fatality ratio']
for i in wek:
    X,y = P.preprocessData(dane, i ,100)
    y_rad=RBF(X,y,100,10)
    wyjscie.append(y_rad[-1])
    #t1 = range(len(y_rad))
    #t2 = range(len(y))
print(wyjscie)
    #print(y_rad[-1])

'''
plt.figure()
plt.plot(t1,y_rad,label="Predykcja")
plt.plot(t2,y,label="Dane oryginalne")
plt.xlabel("Numer próbki")
plt.ylabel(key)
plt.legend()
plt.show()
'''



#P.predix()
#P.gen_data(D)
#P.get_old_data()
#P.predix(D)

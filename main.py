from lib import *
from Scrap import Daily_Raport
from Data_Base import DB
#from Processing import Process


if __name__=="__main__":
    S=Daily_Raport()

    B=DB(S)
    B.insert(S)

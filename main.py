from lib import *
from prog import Daily_Raport
from Data_Base import DB


if __name__=="__main__":
    D=Daily_Raport()
    D.import_data()
    #D.show_raport()
    #D.write_to_file('raports.txt')

    B=DB(D)
    B.insert(D)

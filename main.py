from lib import *
#from Scrap import Daily_Raport
from Data_Base import DB
import win32gui, win32con
from Processing import Process

# the_program_to_hide = win32gui.GetForegroundWindow()
# win32gui.ShowWindow(the_program_to_hide , win32con.SW_HIDE)

#from Processing import Process


if __name__=="__main__":
    D=DB()
    P=Process(D)
    P.plot_new_infected()

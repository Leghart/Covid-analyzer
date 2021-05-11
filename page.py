import win32gui, win32con
from flask import Flask,render_template
import logging
import threading
import time

from data_base import DataBase as DB
from processing import Process as P
# the_program_to_hide = win32gui.GetForegroundWindow()
# win32gui.ShowWindow(the_program_to_hide , win32con.SW_HIDE)


app = Flask(__name__)


def thread_function(name):
    while(True):
        print('siema')
        time.sleep(2)


@app.route("/")
def hmm():
    D=DB('Poland')
    Pl=P(D)
    last_update = D.get_last_record_date()
    return render_template('index.html', **locals())

if __name__ == '__main__':

    #x = threading.Thread(target=thread_function, args=(1,))
    #x.start()
    app.run(debug=True)

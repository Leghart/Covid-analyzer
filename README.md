## Table of contents
* [General info](#general-info)
* [Packages](#Packages)
* [Setup](#setup)
* [Prediction](#prediction)
* [Periodic data download](#periodic-data-download)


## General info
A project aimed at collecting up-to-date data on the coronavirus pandemic and
predicting new cases for the next few days. Additionally, the data collected
with the prediction are automatically placed on the website. In future page
will be probably developed to make easy access from anywhere.


## Packages
Project has created with:
* BeautifulSoup
* SqlAlchemy
* Tensorflow
* Pmdarima
* Flask
* SMTPlib


## Setup
All .py files must be in one directory. Depending on the path to the directory,
change the .bat file (this is described in "Periodic data download").
To run the program, run the main.py file which is responsible for downloading
data and uploading them to the database. The page.py file is responsible for
launching the website. Before used code, you have to prepare setup.py file. There
are a few important fields:
- Country - choose a country to scrap data, insert to
database etc.
- scrap_time - different countries have different upload-data times.
- forecast_hor - how many days you want to predict in future (too high value will
make the network useless)
- db_name - name of the database to be created and connected in the future
where the data will be stored


## Prediction
The following prediction functions have been implemented in this project:
* ARIMA - Autoregressive integrated moving average is trying to forecast a future cases.
ARIMA models are applied in some cases where data show evidence of non-stationarity in
the sense of mean. Actually the best solution which is used. Implemented from pmdarima.
* LSTM - Short-term memory neural network with tensorflow. The network was taught with
75% of the collected data (the remaining 25% is the test kit). The learned model can be
used as simple predictor, but it has a large error, because we don't have enough data
for good learning and new cases depend on things, which I do not collect (e.g. lockdowns
information, possible virus migration, temperature etc). It's just a simple project to
get to know neural networks better).


## Periodic data download
If you want automatically download data you can use: Schedule Manager on Windows
or Corn on Linux. This project was done on Windows so here's a instruction how
to do that:
* Create file .bat and write there:
"full_path_to_python" "full_path_to_Scrap.py"
pause
* Schedule your own date-time to scrap data using a file.bat

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Prediction](#prediction)
* [Periodic data download](#periodic-data-download)

## General info
A project aimed at collecting up-to-date data on the coronavirus pandemic and
predicting new cases for the next few days. Additionally, the data collected
with the prediction are automatically placed on the website.

## Technologies
Project has created with:
* BeautifulSoup
* SqlAlchemy
* Sklearn
* Flask
* SMTPlib

## Setup
All .py files must be in one directory. Depending on the path to the directory,
change the .bat file (this is described in "Periodic data download").
To run the program, run the main.py file which is responsible for downloading
data and uploading them to the database. The page.py file is responsible for
launching the website. Before used code, you have to prepare setup.py file. There
are a few importat fields: 
Country - choose a country to scrap data, insert to 
database etc. 
scrap_time - different countries have different upload-data times.


## Prediction
The following prediction functions have been implemented in this project: 
* RBF - Radial basis function is used for aproximate a function. Uses the self-learning
Kohonen network. 
* ARIMA - Autoregressive integrated moving average is trying to forecast a future cases. 
ARIMA models are applied in some cases where data show evidence of non-stationarity in 
the sense of mean. Actually the best solution which is used. 

## Periodic data download
If you want automatically download data you can use: Schedule Manager on Windows
or Corn on Linux. This project was done on Windows so here's a instruction how
to do that:
* Create file .bat and write there:
"full_path_to_python" "full_path_to_Scrap.py"
pause
* Schedule your own datetime to scrap data using a file.bat

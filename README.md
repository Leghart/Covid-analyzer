## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
It is a Covid analyzer in Poland. It scrap data from gazetawroclawska.pl, add records to data base. In future would
	
## Technologies
Project is created with:
* BeautifulSoup
* SQlite3

## Setup

## Periodic data download
If you want automatically download data you can use: Schedule Manager on Windows or Corn on Linux. This project was done on Windows so here's a instructions how do that:
* Create file .bat and write there: 
"full_path_to_python" "full_path_to_Scrap.py"
pause
* Schedule your own datetime to scrap data (I used to 11 AM) using a file.bat


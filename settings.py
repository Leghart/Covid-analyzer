"""
The main section for changes and personalization for you.
- Country - your country (be careful and check that you enter the name
correctly - if you hesitate, check the names on the
website - www.worldometers.com)
- db_name - name of the database that will be created
- scrap_time - each country has a different data transfer time,
so enter the appropriate one
time (you can enter a little later time for data collection)
- Forecast_hor - number of days to predict in the future (this value cannot be
too large, because all forecasting methods will be affected by error)
"""
import os
from sys import platform

COUNTRY = "Poland"
DB_NAME = "Covid_Data.db"
SCRAP_TIME = "10:00"
FORECAST_HOR = 7
DATE_FORMAT = "%d.%m.%Y"

# Path to directory where files were placed
MAIN_PATH = os.path.dirname(os.path.abspath(__file__))

if platform == "linux":
    OS_CON = "/"
elif platform == "win32":
    OS_CON = "\\"
else:
    print("Not handled yet")

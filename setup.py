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

Country = 'Poland'
db_name = 'Covid_Data.db'
scrap_time = '10:00'
Forecast_hor = 7

from bs4 import BeautifulSoup
from requests import get
from datetime import date
import unicodedata
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

import pylab as pl
import sklearn
import sklearn_evaluation
import math
import warnings

import ssl
import smtplib

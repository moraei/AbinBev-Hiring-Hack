# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:58:28 2018

@author: spallavo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 10:41:17 2018

@author: spallavo
"""

%matplotlib inline

from time import time
from dateutil.relativedelta import *
import datetime as dt
import calendar as calendar
from datetime import datetime
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import os as os
from fbprophet import Prophet
import pyodbc as odbc
import matplotlib.pyplot as plt
import sys as sys
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

plt.style.use('fivethirtyeight')
# Make the graphs a bit prettier, and bigger
#pd.set_option('display.mpl_style', 'default') 
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60) 

data = pd.read_csv('C:/Users/spallavo/Downloads/AbinBev/volume_sales_promox.csv')
data.dtypes

demographics = pd.read_csv('C:/Users/spallavo/Downloads/AbinBev/demographics.csv')
event_calendar = pd.read_csv('C:/Users/spallavo/Downloads/AbinBev/event_calendar.csv')
historical_volume = pd.read_csv('C:/Users/spallavo/Downloads/AbinBev/historical_volume.csv')
industry_soda_sales = pd.read_csv('C:/Users/spallavo/Downloads/AbinBev/industry_soda_sales.csv')
industry_volume = pd.read_csv('C:/Users/spallavo/Downloads/AbinBev/industry_volume.csv')
price_sales_promo = pd.read_csv('C:/Users/spallavo/Downloads/AbinBev/price_sales_promotion.csv')
weather = pd.read_csv('C:/Users/spallavo/Downloads/AbinBev/weather.csv')

historical_volume1 = pd.merge(left=historical_volume, right=price_sales_promo, how='inner', left_on=['Agency','SKU','YearMonth'], right_on=['Agency','SKU','YearMonth'])
historical_volume1 = pd.merge(left=historical_volume1, right=weather, how='inner', left_on=['Agency','YearMonth'], right_on=['Agency','YearMonth'])
historical_volume1 = pd.merge(left=historical_volume1, right=industry_soda_sales, how='inner', left_on=['YearMonth'], right_on=['YearMonth'])
historical_volume1 = pd.merge(left=historical_volume1, right=industry_volume, how='inner', left_on=['YearMonth'], right_on=['YearMonth'])
historical_volume1 = pd.merge(left=historical_volume1, right=demographics, how='inner', left_on=['Agency'], right_on=['Agency'])
historical_volume1 = pd.merge(left=historical_volume1, right=event_calendar, how='inner', left_on=['YearMonth'], right_on=['YearMonth'])

summ_data = historical_volume1.groupby(['Agency','YearMonth','Avg_Max_Temp','Avg_Population_2017','Avg_Yearly_Household_Income_2017','Easter Day','Good Friday','New Year','Christmas','Labor Day','Independence Day','Revolution Day Memorial','Regional Games','FIFA U-17 World Cup','Football Gold Cup','Beer Capital','Music Fest'])['Volume'].sum()
historical_volume2.columns = historical_volume2.columns.droplevel()
historical_volume2.reset_index(inplace=True)

 = 
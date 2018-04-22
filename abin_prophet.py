# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 10:41:17 2018

@author: spallavo
"""

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

historical_volume2 = historical_volume1.groupby([pd.Grouper(key='Agency'),pd.Grouper(key='SKU'),'YearMonth'])[['Volume']].sum().unstack(fill_value=0)
historical_volume2.columns = historical_volume2.columns.droplevel()
historical_volume2.reset_index(inplace=True)

# Move keys into train_key dataframe
train_key = historical_volume2.iloc[:,:2]

#dropping key and moving data into train_key_less
train_key_less = historical_volume2.iloc[:,2:]

N = 1 # number of months for test split
startTime = dt.datetime.now()
all_data = train_key_less.T
#since additional column  nonz is at end of the dataframe, need to offset one additional column 
#pick last N_offset columns and exclude last column as it is nonz column
#offset = 1 #This is representing additional nonz column at end of dataframe
#N_offset = N+offset #This helps to pick last N periods for testing with additional offset of one column that is nonz which should not part of test data, will eliminate from test data
trainxx, testxx = all_data.iloc[0:-N,:], all_data.iloc[-N:,:]
#Above we are picking rows from -N_offset to -1 as last column is nonz t hat is eliminated from test data set
test_cleaned = testxx.T.fillna(method='ffill').T
train_cleaned = trainxx.T.iloc[:,:].fillna(method='ffill').T
df_final = pd.DataFrame([])
#sample_train_cleaned = train_cleaned.iloc[:,1:41].T

#i=51

startTime = dt.datetime.now()
for i in range(0,10): #train_cleaned.shape[1] is the number of IPNs, so we have to forecast each one at a time
#    #fill outliers that are out of 1.5*std with rolling mean of 10 days
    df_one_agency_data = train_cleaned.iloc[:,i].to_frame()
    df_one_agency_data.columns = ['forecasted_vol']
#    #Calculate mean for outlier treatment
#    df_one_agency_data['mean'] = round(df_one_agency_data.forecasted_vol.rolling(window=10,min_periods=1,center=False).mean(), 2)
#    std_mult = 1.5
#    #outlier treatment - If difference of actual value from mean is more than 1.5 times of STDEV, then assign mean as forecasted_vol value
#    df_one_agency_data.ix[np.abs(df_one_agency_data.forecasted_vol - df_one_agency_data.forecasted_vol.mean())>=(std_mult*df_one_agency_data.forecasted_vol.std()),'forecasted_vol'] = df_one_agency_data.ix[np.abs(df_one_agency_data.forecasted_vol-df_one_agency_data.forecasted_vol.mean())>=(std_mult*df_one_agency_data.forecasted_vol.std()),'mean']
    # Converting the index as date
    df_one_agency_data.index = pd.to_datetime(df_one_agency_data.index, format='%Y%m', errors='coerce')
    
    test_cleaned

    #prophet expects the folllwing label names
    X = pd.DataFrame(index=range(0,len(df_one_agency_data)))
    X['ds'] = df_one_agency_data.index
    X['y'] = df_one_agency_data['forecasted_vol'].values
    #X['y'] = np.log(X['y'] + 1)
    X
    #Plot the data and see how it looks like
    #my_plt = X.set_index('ds').plot(figsize=(12, 8))
    
    m = Prophet()
    if sum(X['y']) != 0:
        m.fit(X)
        #print("No")
        future = m.make_future_dataframe(periods=1, freq='MS', include_history=False) #Here MS is start of Month for frequency in prophet method
        forecast = m.predict(future)
        #get prediction results for future test data for which timeseries is available in df_future
        df_future_forecast = forecast[["ds", "yhat_upper", "yhat", "yhat_lower"]]
        
        df_forecast_new = pd.concat([train_key.iloc[i:i+1,].reset_index(drop=True), df_future_forecast.loc[:,['ds','yhat']].reset_index(drop=True)], axis=1, ignore_index=True)
        
        df_test_data_y = test_cleaned.T.iloc[i:i+1,0:1].copy()
        df_yhat_y = pd.concat([df_forecast_new.reset_index(drop=True), df_test_data_y.reset_index(drop=True)], axis=1, ignore_index=True)
        #final = pd.concat([final, df_yhat_y])
        df_yhat_y.columns = ['Agency','SKU', 'ds','yhat','y']
        
        df_final = df_final.append(df_yhat_y, ignore_index=True)
    else:
        print("Skipping forecast for IPN:"+ str(train_key.iloc[i:i+1,0]))
    print(i)


# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:57:42 2018

@author: vprayagala2
"""
#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#%%
def load_data():
#read data from files
    #1. read hist volume data
    data=pd.read_csv("./train/historical_volume.csv")
    sales_prom=pd.read_csv("./train/price_sales_promotion.csv")
    data=pd.merge(data,sales_prom,
                   on=['Agency','SKU','YearMonth'],
                       )
    
    ind_vol=pd.read_csv("./train/industry_volume.csv")
    data=pd.merge(data,ind_vol,
                   on=['YearMonth'],
                       )
    ind_soda_sales=pd.read_csv("./train/industry_soda_sales.csv")
    data=pd.merge(data,ind_soda_sales,
                   on=['YearMonth'],
                       )
    
    agency_demo=pd.read_csv("./train/demographics.csv")
    data=pd.merge(data,agency_demo,
                   on=['Agency'],
                       )
    agency_wth=pd.read_csv("./train/weather.csv")
    data=pd.merge(data,agency_wth,
                   on=['Agency','YearMonth'],
                       )
    
    event_cal=pd.read_csv("./train/event_calendar.csv")
    data=pd.merge(data,event_cal,
                   on=['YearMonth'],
                       )
    return data,sales_prom,ind_vol,ind_soda_sales,agency_demo,agency_wth,event_cal

def process_data(data,columns_to_drop=['YearMonth','Volume']):
    data_target=data.loc[:,['Volume']]
    data.drop(columns_to_drop,axis=1,inplace=True)

    le_agency=LabelEncoder()
    le_sku=LabelEncoder()
    data['Agency']=le_agency.fit_transform(data['Agency'])
    data['SKU']=le_sku.fit_transform(data['SKU'])

    return data,data_target,le_agency,le_sku
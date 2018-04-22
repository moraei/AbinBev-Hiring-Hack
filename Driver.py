# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:56:51 2018

@author: vprayagala2
Forecasting Driver Routine - The forecasting method to be used,the predictions 
to be made have to be input
"""
#%%
#Set the utility path
import sys,os
sys.path.append(os.getcwd())
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
#%%
#Import the packages
import pandas as pd
import numpy as np
import ForecastWrapper as FW
#Set Logger
#import time
from sklearn.model_selection import train_test_split
from time import strftime
import CreateLogger
import DataProcessor as DP
#%%
log_file="Log_"+(strftime("%Y_%m_%d_%H_%M_%S"))+".txt" 
logger=CreateLogger.create_logger(log_file)
data,sales_prom,ind_vol,ind_soda_sales,agency_demo,agency_wth,event_cal=DP.load_data()
columns_to_drop=['YearMonth','Volume','Price','Sales','Promotions']
md_data,md_targ,le_agency,le_sku=DP.process_data(data,columns_to_drop)
split_size=0.2
seed=77
X_train,X_test,Y_train,Y_test=train_test_split(md_data,md_targ, \
                                test_size=split_size,random_state=seed)
TrainX=X_train.values
TrainY=Y_train.values
TestX=X_test.values
TestY=Y_test.values
fcst_obj=FW.ForecastWrapper(logger)
#%%
Model_Res=fcst_obj.build_regression_model(TrainX,TrainY.ravel(),
                                          TestX,TestY.ravel())
Model_Ens=fcst_obj.build_regression_ensembles(TrainX,TrainY.ravel(),
                                              TestX,TestY.ravel())
Result=pd.DataFrame(Model_Res,index=['LR','CART','SVM'])
Result=Result.append(pd.DataFrame(Model_Ens,index=['RF','AB','GBM']))
Result.columns=['Train_RMSE','Test_RMSE']
#pred_prob=RF_Model.predict_proba(Xtest_scaled)
Result.plot()
#%%
#Fine tune RF with grid search
Model=fcst_obj.tune_model(TrainX,TrainY.ravel(),TestX,TestY.ravel())
#%%
#Forecast the timebased attributes
#Combine timebased and non-time based attributes for model and then regress the outout
ind_vol.columns = ['ds','y']
ind_vol_ts=fcst_obj.process_ts_data(ind_vol)
ts_mdl1=fcst_obj.build_model(ind_vol_ts) 
future,forecast=fcst_obj.make_predictions(ts_mdl1,ind_vol_ts,period=1,freq='MS')
ind_vol_fut=forecast[-1:]['yhat']
#%%
ind_soda_sales.columns=['ds','y']
ind_soda_ts=fcst_obj.process_ts_data(ind_soda_sales)
ts_mdl2=fcst_obj.build_model(ind_soda_ts) 
future,forecast=fcst_obj.make_predictions(ts_mdl2,ind_soda_ts,period=1,freq='MS')
ind_soda_fut=pd.DataFrame()
ind_soda_fut=forecast[-1:]['yhat']
#%%
Ag_List=np.unique(agency_wth['Agency'])
num_ag=len(Ag_List)
dict_of_df={}
res_df=pd.DataFrame()
for i,Agent in enumerate(Ag_List):
    print("Extracting Data For {} ".format(Agent))
    key_name=Agent
    dict_of_df[key_name]=agency_wth.loc[agency_wth.Agency == Agent
             ,['YearMonth','Avg_Max_Temp']]
    dict_of_df[key_name].columns=['ds','y']
    data_processed=fcst_obj.process_ts_data(dict_of_df[key_name])
    ts_mdl=fcst_obj.build_model(data_processed) 
    future,forecast=fcst_obj.make_predictions(ts_mdl,data_processed,period=1,freq='MS')
    df=pd.DataFrame()
    df['Avg_Max_Temp']=forecast[-1:]['yhat']
    df['Agency']=Agent
    res_df=pd.concat([res_df,df],axis=0)
      
#%%
#Read Test Data , Append time and non-time features to it
#then use model to predict volume
test_data=pd.read_csv("./test/volume_forecast.csv")
test_data.drop(['Volume'],axis=1,inplace=True)

test_data['Industry_Volume']=pd.Series(ind_vol_fut.values,index=test_data.index)
test_data['Soda_Volume']=pd.Series(ind_soda_fut.values,index=test_data.index)
test_data=pd.merge(test_data,agency_demo,
                   on=['Agency'])
test_data=pd.merge(test_data,res_df,
                   on=['Agency'])

event=event_cal[-1:]
event=event.iloc[:,1:]
num=len(test_data.index)
event=pd.concat([event]*num, ignore_index=True)
test_data=pd.concat([test_data,event],axis=1)

test_data['Agency']=le_agency.fit_transform(test_data['Agency'])
test_data['SKU']=le_sku.fit_transform(test_data['SKU'])
Volume=Model.predict(test_data)
#%%
test_data['Volume']=Volume
final_result=test_data[['Agency','SKU','Volume']]
final_result['Agency']=le_agency.inverse_transform(final_result['Agency'].values)
final_result['SKU']=le_sku.inverse_transform(final_result['SKU'].values)
with open("./submission/volume_forecast.csv","w") as f:
    final_result.to_csv(f,index=False)
#%%
#Predict volume for Agency 6 and 14 for all SKUs and pick best two
def sku_recomm(agency):
    test_data_1=pd.DataFrame()
    test_data_1['SKU']=pd.Series(np.unique(md_data['SKU']))
    test_data_1['Agency']=agency
    
    test_data_1['Industry_Volume']=pd.Series(ind_vol_fut.values,index=test_data_1.index)
    test_data_1['Soda_Volume']=pd.Series(ind_soda_fut.values,index=test_data_1.index)
    test_data_1=pd.merge(test_data_1,agency_demo,
                       on=['Agency'])
    test_data_1=pd.merge(test_data_1,res_df,
                       on=['Agency'])
    
    test_data_1['Agency']=le_agency.fit_transform(test_data_1['Agency'])
    event=event_cal[-1:]
    event=event.iloc[:,1:]
    num=len(test_data_1.index)
    event=pd.concat([event]*num, ignore_index=True)
    test_data_1=pd.concat([test_data_1,event],axis=1)
    Volume=Model.predict(test_data_1)

    test_data_1['Volume']=Volume
    final_result1=test_data_1[['Agency','SKU','Volume']]
    final_result1['Agency']=le_agency.inverse_transform(final_result1['Agency'].values)
    final_result1['SKU']=le_sku.inverse_transform(final_result1['SKU'].values)
    final_result1.sort_values('Volume',ascending=False,inplace=True)
    final_result1=final_result1.iloc[0:2,:]
    return final_result1
#%%
res1=sku_recomm('Agency_06')
res2=sku_recomm('Agency_14')
res=pd.concat([res1,res2],ignore_index=True)
final_result1=res[['Agency','SKU']]
#%%
with open("./submission/sku_recommendation.csv","w") as f:
    final_result1.to_csv(f,index=False)










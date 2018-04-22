# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:57:46 2017

@author: vprayagala2

Utility Class for Prophet - More methods to be added

##############################################################################
#### Revision LOG    #########################################################
# Version   Date            User            Comments
# 0.1       10/22/2017      TA ML Team      Initial Draft
##############################################################################
"""
#%%
#Load the required Libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from fbprophet import Prophet
import pickle
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.seasonal import seasonal_decompose  
import math
from sklearn.cross_validation import KFold 
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#%%
#Get the tran data
class ForecastWrapper:
    def __init__(self,logger):
        #self.logger=logging.getLogger(__name__)
        self.logger=logger
        
       
    def process_ts_data(self,data):
        data['ds']=pd.to_datetime(data['ds'],format='%Y%m')
        data['y']=pd.to_numeric(data['y'],errors='coerce')
        self.logger.info(data.head())
        columns_to_drop=[column for column in data.columns if column not in ['ds','y']]
        data.drop(columns_to_drop,inplace=True,axis=1)
        
        #data.columns=['ds','y']
        self.logger.info(data.head())
        self.logger.info(data.dtypes)
        return data
    
      
    def build_model(self,data,seasonality='None',interval_width=0.8,changepoint_prior_scale=0.07):
        model=Prophet(interval_width=interval_width,\
                      changepoint_prior_scale=interval_width)
        if seasonality not in ['Yearly','Monthly','Quarterly','Weekly','Daily','Hourly','None']:
            print("Invalid Seasonality Parameter:{}".format(seasonality))
            self.logger.error("Invalid Seasonality Parameter:{}".format(seasonality))
            return None
        else:
            if seasonality == 'Yearly':
                model.add_seasonality(name='Yearly',period=365,fourier_order=7)
            if seasonality == 'Monthly':
                model.add_seasonality(name='Monthly',period=30,fourier_order=7)  
            if seasonality == 'Quarerly':
                model.add_seasonality(name='Quarterly',period=120,fourier_order=7)
            if seasonality == 'Weekly':
                model.add_seasonality(name='Weekly',period=7,fourier_order=7)
            if seasonality == 'Daily':
                model.add_seasonality(name='Daily',period=1,fourier_order=7)
            if seasonality == 'Hourly':
                model.add_seasonality(name='Hourly',period=int(1/24),fourier_order=7)
            model.fit(data)
            return model
    
    def build_model_with_holidays(self,data,seasonality,holidays):
        model=Prophet(growth='logistic',\
                      interval_width=0.95,\
                      n_changepoints=8,\
                      changepoint_prior_scale=0.7,\
                      holidays=holidays)
        model.add_seasonality(name='hourly',period=8,fourier_order=7)
        model.fit(data)
        return model        
        
    def make_predictions(self,model,data,period=72,freq='min'):
        self.logger.info("Forecast Frequency %s"%(freq))
        future=model.make_future_dataframe(periods=int(period),freq=freq)
        forecast=model.predict(future)
        return future,forecast
    
    def save_model(self,model,model_file):
        with open(model_file,"wb") as f:
            pickle.dump(model,f)
            
    def read_model(self,model_file):
        with open(model_file,"rb") as f:
            model=pickle.load(f)            
            return model
        
    def test_stationarity(self,timeseries,rolling_window=12):
        
        #Determing rolling statistics
        rolmean = pd.Series.rolling(timeseries, window=rolling_window).mean()
        rolstd = pd.Series.rolling(timeseries, window=rolling_window).std()
    
        #Plot rolling statistics:
        #plt.plot(timeseries, color='blue',label='Original')
        #plt.plot(rolmean, color='red', label='Rolling Mean')
        #plt.plot(rolstd, color='black', label = 'Rolling Std')
        #plt.legend(loc='best')
        #plt.title('Rolling Mean & Standard Deviation')
        #plt.show(block=False)
        
        #Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)    


#    def decompose_ts(self,ts,decom_freq):
#        decomposition = seasonal_decompose(ts,freq=decom_freq)
#    
#        trend = decomposition.trend
#        seasonal = decomposition.seasonal
#        residual = decomposition.resid
#    
#        plt.subplot(411)
#        plt.plot(ts, label='Original')
#        plt.legend(loc='best')
#        plt.subplot(412)
#        plt.plot(trend, label='Trend')
#        plt.legend(loc='best')
#        plt.subplot(413)
#        plt.plot(seasonal,label='Seasonality')
#        plt.legend(loc='best')
#        plt.subplot(414)
#        plt.plot(residual, label='Residuals')
#        plt.legend(loc='best')
#        plt.tight_layout()
        
    def calc_rmse_error(self,y_true,y_pred):
        rmse_error=math.sqrt(mean_squared_error(y_true, y_pred))
        return rmse_error
    
    def tune_model(self,TrainX,TrainY,TestX,TestY):
        #Use Gridsearch to tune mode, TBD
        n_est=[50,40,30,20,10]
        max_feat=['auto','sqrt','log2']
        param_grid = dict(n_estimators=n_est,\
                          max_features=max_feat
                          )
        model = RandomForestRegressor()
        kfold = KFold(n=TrainX.shape[0],n_folds=10, random_state=77)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold)
        grid_result = grid.fit(TrainX, TrainY)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        RF_Model=RandomForestRegressor(max_features=grid_result.best_params_['max_features'],\
                                n_estimators=grid_result.best_params_['n_estimators']
                                )

        
        RF_Model.fit(TrainX,TrainY)      
        return RF_Model
    
    
    def build_regression_model(self,TrainX,TrainY,TestX,TestY):
        
        models=[]
        models.append(('LR',LinearRegression()))
        models.append(('CART',DecisionTreeRegressor()))
        models.append(('SVM',SVR()))
        
        results=[]
        names=[]
        for name, model in models:
            model.fit(TrainX,TrainY)
            y_pred=model.predict(TrainX)
            rmse_train=self.calc_rmse_error(TrainY,y_pred)
            y_pred=model.predict(TestX)
            rmse_test=self.calc_rmse_error(TestY,y_pred)
            results.append([rmse_train,rmse_test])
            names.append(name)
            msg = "%s: %f %f" % (name, rmse_train, rmse_test)
            print(msg)
        return results
            
    def build_regression_ensembles(self,TrainX,TrainY,TestX,TestY):
        #Try with ensembles
        ensembles = []
        ensembles.append(('RF', RandomForestRegressor()))
        ensembles.append(('AB', AdaBoostRegressor()))
        ensembles.append(('GBM', GradientBoostingRegressor()))
        
        
        results = []
        names = []
        for name, model in ensembles:
            model.fit(TrainX,TrainY)
            y_pred=model.predict(TrainX)
            rmse_train=self.calc_rmse_error(TrainY,y_pred)
            y_pred=model.predict(TestX)
            rmse_test=self.calc_rmse_error(TestY,y_pred)
            results.append([rmse_train,rmse_test])
            names.append(name)
            msg = "%s: %f %f" % (name, rmse_train, rmse_test)
            print(msg)
        return results

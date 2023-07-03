# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:46:23 2020

@author: Administrator
"""
import numpy as np
from numpy import asarray
import pandas as pd
from pandas import DataFrame
from pandas import concat
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    n_vars = data.shape[1]
    df = DataFrame(data)
    cols, names = list(),list()
    for i in range(n_in, 0 ,-1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names+=[('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names+= [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace = True)
    return agg

def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

def xgboost_forecast(train, testX):
    train = asarray(train)
    trainX, trainy = train[:,:-1], train[:,-1]

    model = XGBRegressor(objective = 'reg:squarederror', n_estimators = 1000)
    model.fit(trainX, trainy)
    yhat = model.predict(testX)
    return yhat[0]


def walk_forward_validation(data, n_test):
    predictions = list()
    train, test = train_test_split(np.array(data), n_test)
    history = [x for x in train]
    for i in range(len(test)):
        testX, testy = test[i, :-1], test[i, -1]
        yhat = xgboost_forecast(history, testX)
        predictions.append(yhat)
        history.append(yhat)
        print('expected:%.2f, predicted=%.2f' % (testy. yhat))
    maeerror = mean_absolute_error(test[:,1], predictions)
    mseerror = mean_squared_error(test[:,1], predictions)
    return maeerror, mseerror, test[:,1], predictions

values = pd.read_csv('dataset2.csv')
data = series_to_supervised(values, 12)#, 12)
#print(data)
mae, mse, y, yhat = walk_forward_validation(data, 12)

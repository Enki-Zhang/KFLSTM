# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:46:43 2020

@author: Administrator
"""
# forecast monthly births with xgboost
import time
import numpy as np
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from matplotlib import pyplot

time_start = time.time()


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    print("data",data.shape)
    print("data[:-n_test, :]",data[:-n_test, :].shape)
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    # model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model = XGBRegressor( max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0]


# walk-forward validation for univariate data 单变量预测
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    test_copy = test.copy()
    print("train数据",type(train),train.size,train.shape)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        test[i][35] = yhat
        history.append(test[i])
        # summarize progress
        print('>expected=%.2f, predicted=%.2f' % (testy, yhat))
    # estimate prediction error
    predictions = np.array(predictions)
    maeerror = mean_absolute_error(test_copy[:, -1], predictions)
    mseerror = mean_squared_error(test_copy[:, -1], predictions)
    rmseerror = np.sqrt(mseerror)
    # nrmse#
    t1 = np.sum((test_copy[:, -1] - predictions) ** 2) / np.size(predictions)
    t2 = np.sum(abs(predictions)) / np.size(predictions)
    nrmseerror = np.sqrt(t1) / t2

    return maeerror, mseerror, rmseerror, nrmseerror, test_copy[:, -1], predictions


# load the dataset
# series = read_csv('海门湾2000x5.csv')
# series = read_csv('dataset2测试Tem.csv')
# series = read_csv('原始数据填0测试Tem.csv')
series = read_csv('海门湾label=4(0).csv')
# series = read_csv('dataset2.csv')
series = series.fillna(0)
values = series.values
print("values",values.shape)
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=96, n_out=48)
print("values->data",data.shape)
# evaluate
mae, mse, rmse, nrmse, y, yhat = walk_forward_validation(data, 1000)
print('MAE: %.3f' % mae)
print('\nMSE: %.3f' % mse)
print('\nRMSE: %.3f' % rmse)
print('\nNRMSE: %.3f' % nrmse)

# plot expected vs preducted
# pyplot.figure(figsize=(24, 4))
pyplot.plot(y[:48], label='Expected', alpha=0.3)
pyplot.plot(yhat[:48], label='Predicted')
pyplot.legend()
pyplot.show()
time_end = time.time()
print('time consumption:', time_end - time_start)
np.savetxt('XGBLabel.csv', y)
np.savetxt('XGBPre.csv', yhat)
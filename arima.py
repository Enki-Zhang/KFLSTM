# import pandas as pd
# import  pandas_datareader
# import datetime
# import matplotlib.pylab as plt
# import seaborn as sns
# from matplotlib.pylab import style
# from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# style.use('ggplot')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# stockFile = '海门湾带时间列（缺）.csv'
# stock = pd.read_csv(stockFile, index_col= 0, parse_dates = [0]).fillna(0)
# stock.index = pd.to_datetime(stock.index)
# # print(stock.index)
# # print(stock.tail(10))
# stock_train = stock['oxy'][:7877]
# stock_test = stock['oxy'][7877:]
# # # stock_train.plot(figsize = (22,8))
# # #stock_test.plot(figsize = (22,8))
# # #plt.legend(bbox_to_anchor=(1.25,0.5))
# # #sns.despine()

# stock_diff = stock_train.diff()
# #stock_diff = stock_diff.diff()
# stock_diff = stock_diff.dropna()

# # plt.figure()
# # plt.plot(stock_diff)
# plt.show()

# # acf = plot_acf(stock_diff, lags = 20)
# # plt.title('ACF')
# # acf.show()

# # pacf = plot_pacf(stock_diff, lags = 20)
# # plt.title("PACF")
# # pacf.show()

# model = ARIMA(stock_train, order=(1,1,1))

# result = model.fit()
# pred = result.predict('2020-06-01 00:00:00',
# '2020-06-30 00:00:00', dynamic=True, typ='levels')
# print(pred)
import os, sys
# os.chdir(sys.path[0])
import pandas as pd
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm
import numpy as np
import time

time_start = time.time()


def parser(x):
    return pd.datetime.strptime(x, '%Y/%m/%d %H:%M')


# def parser(x):
#     return datetime.strptime(x, '%Y/%m/%d %H:%M')
# series = read_csv('海门湾带时间列.csv', header=0, parse_dates=[0], 
#                   index_col=0, squeeze=True, date_parser=parser)
# series = pd.read_csv('./KRMData/oxy/海门湾label=4.csv')
series = pd.read_csv('dataset2.csv')
series = series.fillna(0)

# series = series.fillna(method = 'ffill', inplace=False)
# X = series['oxy'].values
oxy = series['Dissolved oxygen']
oxy.plot()
diff = (oxy.diff(1))
diff.plot()
fig = plt.figure(figsize=(30, 8))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(diff, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(diff, lags=40, ax=ax2)
plt.show()


def nrmse(y_pred, y_true):
    """ Normalized RMSE"""
    t1 = np.sum((y_pred - y_true) ** 2) / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return np.sqrt(t1) / t2


epochs = (len(oxy) // 36)
error_sum1 = 0
error_sum2 = 0
error_sum3 = 0
error_sum4 = 0
step = 325
for i in range(epochs):

    input = step + 24
    train, test = oxy[step:input], oxy[input:input + 12]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = sm.tsa.arima.ARIMA(history, order=(1, 2, 1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[input + t]
        history.append(float(yhat))
        print('predicted=%f, expected=%f, his:%f' % (yhat, obs, len(history)))
    step = step + 36
    error1 = mean_absolute_error(np.array(test), predictions)
    error2 = mean_squared_error(np.array(test), predictions)
    error3 = np.sqrt(error2)
    error4 = nrmse(np.array(test), np.array(predictions))
    # print(error1)
    error_sum1 = error_sum1 + error1
    error_sum2 = error_sum2 + error2
    error_sum3 = error_sum3 + error3
    error_sum4 = error_sum4 + error4
    time_end = time.time()

mae = error_sum1 / epochs
mse = error_sum2 / epochs
nrmse = error_sum4 / epochs
print('Test MSE: %.3f' % mse, 'Test NRMSE: %.3f' % nrmse)
# plot
# plt.plot(test)
# plt.plot(predictions, color='red')

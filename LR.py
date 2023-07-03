# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:34:55 2020

@author: Administrator
"""
from __future__ import print_function

# 导入相关python库
import os
import numpy as np
import pandas as pd

# 设定随机数种子
np.random.seed(36)

# 使用matplotlib库画图
import matplotlib
import seaborn
import matplotlib.pyplot as plot
import time
from sklearn import datasets

time_start = time.time()
# 读取数据
rowdata = pd.read_csv('dataset2.csv')
rowdata0 = np.array(rowdata.fillna(0))
rowdata1 = np.array(rowdata.fillna(method='ffill', inplace=False))

# def minmaxscale(data: np.ndarray):
#     seq_len, num_features = data.shape
#     for i in range(num_features):
#         min = data[:, i].min()
#         max = data[:, i].max()
#         data[:, i] = (data[:, i] - min) / (max - min)
#     return data

# rowdata0 = minmaxscale(rowdata0)
# rowdata1 = minmaxscale(rowdata1)

train_fea = rowdata0[0:800, 0:4]
train_label = rowdata1[0:800, 4]

test_fea = rowdata0[800:1000, 0:4]
test_label = rowdata1[800:1000, 4]


# 数据预处理
# housing.info()    #查看是否有缺失值

# #特征缩放
# from sklearn.preprocessing import MinMaxScaler
# minmax_scaler=MinMaxScaler()
# minmax_scaler.fit(housing)   #进行内部拟合，内部参数会发生变化
# scaler_housing=minmax_scaler.transform(housing)
# scaler_housing=pd.DataFrame(scaler_housing,columns=housing.columns)

# mm=MinMaxScaler()
# mm.fit(t)
# scaler_t=mm.transform(t)
# scaler_t=pd.DataFrame(scaler_t,columns=t.columns)
def nrmse(y_pred, y_true):
    """ Normalized RMSE"""
    t1 = np.sum((y_pred - y_true) ** 2) / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return np.sqrt(t1) / t2


def R(Pred, Label):
    SStot = np.sum((Label - np.mean(Pred)) ** 2)
    SSres = np.sum((Label - Pred) ** 2)
    r2 = 1 - SSres / SStot
    return r2


# 选择基于梯度下降的线性回归模型
from sklearn.linear_model import LinearRegression

LR_reg = LinearRegression()
# 进行拟合
LR_reg.fit(train_fea, train_label)

# 使用均方误差用于评价模型好坏
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

preds = LR_reg.predict(test_fea)  # 输入数据进行预测得到结果
mae = mean_absolute_error(preds, test_label)
mse = mean_squared_error(preds, test_label)
rmse = np.sqrt(mse)
nrmse = nrmse(preds, test_label)
r = R(preds, test_label)
# 使用均方误差来评价模型好坏，可以输出mse进行查看评价值
print('mae:%.2f' % mae, 'mse:%.2f' % mse, 'rmse:%.2f' % rmse, 'nrmse:%.2f' % nrmse, 'nrmse:%.2f' % r)

# 绘图进行比较
plot.figure(figsize=(8, 6))  # 画布大小
num = 200
x = np.arange(1, num + 1)  # 取100个点进行比较
plot.plot(x, test_label[:num], label='target')  # 目标取值
plot.plot(x, preds[:num], label='preds')  # 预测取值
plot.legend(loc='upper right')  # 线条显示位置
plot.show()
time_end = time.time()
print(time_end - time_start)

# 输出测试数据
# result=LR_reg.predict(scaler_t)
# df_result=pd.DataFrame(result)
# df_result.to_csv("result.csv")

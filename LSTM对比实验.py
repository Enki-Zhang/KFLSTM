# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:27:43 2020

@author: Administrator
"""
import numpy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import time

from sklearn import preprocessing

time_start = time.time()

BATCH_SIZE = 5
LR = 0.001
EPOCH = 10
NUM_LAYERS = 3
HIDDEN_SIZE = 128

# df = pd.read_csv('HMWSmall.csv')
# df = pd.read_csv('dataset2.csv')
df = pd.read_csv('海门湾label=4.csv')
# df = pd.read_csv('海门湾label=4.csv')
# rowdata1 = np.array(df.fillna(0))

rowdata1 = np.array(df.fillna(method='ffill', inplace=False))


# df = df.fillna(method='ffill',inplace = False)
# rowdata = np.array(df)
# 特征归一化
def minmaxscale(data: np.ndarray):
    seq_len, num_features = data.shape
    for i in range(num_features):
        min = data[:, i].min()
        max = data[:, i].max()
        data[:, i] = (data[:, i] - min) / (max - min)
        print("最小最大值")
        print(min, max, (max - min))
    return data


def minmax(data: np.ndarray):
    min = data[:].min()
    max = data[:].max()
    # data[:] = (data[:] - min) / (max - min)
    print("最小最大值")
    print(min, max, (max - min))


trainsize = int(0.8 * 13128)
rowdata1 = minmaxscale(rowdata1)
# # true_label = rowdata1[0:trainsize, 4]]
# train = rowdata1[0:trainsize, 4]
# minmax(train)
train_data = torch.FloatTensor(rowdata1[0:trainsize, 0:4])
train_label = torch.FloatTensor(rowdata1[0:trainsize, 4])

test_data = torch.FloatTensor(rowdata1[trainsize:, 0:4])
test_label = torch.FloatTensor(rowdata1[trainsize:, 4])

dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feat_embed = torch.nn.Linear(1, HIDDEN_SIZE)  # 全连接层 1为 输入二维张量的为1 HIDDEN_SIZE 为输出大小 全连接层的神经元个数。

        self.hidlay = torch.nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                                    batch_first=True)

        # try rnn model
        # self.hidlay = torch.nn.RNN(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        #                            batch_first=True)

        # try GRU model
        # self.hidlay = torch.nn.GRU(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        #                            batch_first=True)

        self.outlay = torch.nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = self.feat_embed(x)
        r_out, (h_n, h_c) = self.hidlay(x, None)  # this is for LSTM model
        # r_out, h_n = self.hidlay(x, None)  # this is for RNN model
        # r_out, h_n = self.hidlay(x, None)  # this is for GRU model
        out = self.outlay(r_out[:, -1, :])  # 获取最后一个时间步输出
        return out


mod = Model()
print("mode")
print(mod)

optimizer = torch.optim.Adam(mod.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 4, 1)
        output = mod(b_x)
        b_y = torch.unsqueeze(b_y, dim=1)
        # print("时间步维度",output.size())
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print('EPOCH:', epoch, 'train loss: %.4f' % loss.data.numpy())


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


optimizer.zero_grad()

# result = mod(test_data.view(-1, 4, 1)) * 13.95 + 0.8  # 反归一化
result = mod(test_data.view(-1, 4, 1)) * 14.95+0.05-4
# mean = result.mean()
# test_label = torch.unsqueeze(test_label, dim=1) * 13.95 + 0.8
test_label = torch.unsqueeze(test_label, dim=1) * 14.95+0.05
# result = result[-2000:,:]
# test_label = test_label[-2000:,:]
# loss_test1 = loss_func(result, test_label).detach().numpy()
a = result.detach().numpy()  # 预测值
b = test_label.detach().numpy()
# print("对比值", b.shape)
# print(a[-100:])
# print("-------------------")
# print(b[-100:])
# a = a[-100:]
# print("预测结果100 A", a)
# b = b[-100:]
# numpy.savetxt("对比实验GRU.csv", b)
# print("预测结果100 B", b)
# plt.figure(figsize=(24, 4))
plt.plot(b[-100:], label="true")
plt.plot(a[-100:], label="pre")
plt.legend()
plt.show()
# loss_test1 = mean_absolute_error(a, b)
loss_test1 = mean_absolute_error(a, b)
loss_test2 = mean_squared_error(a, b)
loss_test3 = np.sqrt(loss_test2)
loss_test4 = nrmse(a, b)
r = R(a, b)
# loss_mean = loss_func(mean, test_label)
time_end = time.time()
time_consumption = time_end - time_start
print('mae:%.2f' % loss_test1, 'mse:%.2f' % loss_test2, 'rmse:%.2f' % loss_test3, 'nrmse: %.2f' % loss_test4,
      'R: %.2f' % r,
      '\ntime consumption:%.2f' % time_consumption)

# xxx = np.array(pd.read_csv('xxx.csv'))
# # xxx[:,0] = (xxx[:,0]-16.7)/5.2
# # xxx[:,1] = (xxx[:,1]-7.05)/1.05
# # xxx[:,2] = (xxx[:,2]-141)/5219
# # xxx[:,3] = (xxx[:,3]-1)/499
#
# zzz = np.array(pd.read_csv('zzz.csv'))
# zzzhat = np.array(pd.read_csv('zzzhat.csv'))
#
# xx = torch.FloatTensor(xxx)
# zz = mod(xx.view(-1, 4, 1))
# zz = zz.detach().numpy() * 10.68
# plt.figure(2)
# plt.plot(zz, color='#1ABC9C', label='FC-LSTM')
# plt.plot(zzz, color='#616A6B', label='Target')
# plt.plot(zzzhat, color='#F1948A', label='FM-GRU')
# plt.xlabel('time step')
# plt.ylabel('value')
# plt.legend()
# plt.show()
# zzloss = mean_squared_error(zz, zzzhat)
# print(zzloss)

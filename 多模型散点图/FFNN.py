# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:16:37 2020

@author: Administrator
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import time

time_start = time.time()

BATCH_SIZE = 128
LR = 0.5
EPOCH = 10
NUM_LAYERS = 2
HIDDEN_SIZE = 128

# df = pd.read_csv('./dataset2.csv')
df = pd.read_csv('海门湾label=4(0).csv')
rowdata0 = np.array(df.fillna(0))
rowdata1 = np.array(df.fillna(method='ffill', inplace=False))


# df = df.fillna(method='ffill',inplace = False)
# rowdata = np.array(df)


def minmaxscale(data: np.ndarray):
    seq_len, num_features = data.shape
    for i in range(num_features):
        min = data[:, i].min()
        max = data[:, i].max()
        data[:, i] = (data[:, i] - min) / (max - min)
    return data


rowdata0 = minmaxscale(rowdata0)
rowdata1 = minmaxscale(rowdata1)

# train_data = torch.FloatTensor(rowdata0[0:10000,0:4])
# train_label = torch.FloatTensor(rowdata1[0:10000,4])
#
# test_data = torch.FloatTensor(rowdata0[10000:,0:4])
# test_label = torch.FloatTensor(rowdata1[10000:,4])
#
trainsize = int(0.8 *13128)
train_data = torch.FloatTensor(rowdata1[0:trainsize, 0:4])
train_label = torch.FloatTensor(rowdata1[0:trainsize, 4])

test_data = torch.FloatTensor(rowdata1[trainsize:, 0:4])
test_label = torch.FloatTensor(rowdata1[trainsize:, 4])
# test_label = rowdata1[800:1000, 4]

dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.inputlay = torch.nn.Linear(input_size, hidden_size)  # inputsize 4
        self.relu = torch.nn.ReLU()
        self.outlay = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.inputlay(x)
        x = self.relu(x)
        out = self.outlay(x)
        return out


mod = Model(4, HIDDEN_SIZE, 1)
# mod.cuda()
print(mod)

optimizer = torch.optim.Adam(mod.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        # b_x = b_x.view(-1,4,1)
        output = mod(b_x)
        b_y = torch.unsqueeze(b_y, dim=1)
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


optimizer.zero_grad()

result = mod(test_data)+7.05 #* 1.05 + 7.05
# result = mod(test_data)
print("label",test_label)
test_label = torch.unsqueeze(test_label, dim=1)+7.05 #* 1.05 + 7.05
# test_label = torch.unsqueeze(test_label, dim=1)
# result = result[-2000:,:]
# test_label = test_label[-2000:,:]
# loss_test1 = loss_func(result, test_label).detach().numpy()
a = result.detach().numpy()
np.savetxt('FFNNResult.csv', a)

print(a.shape)
print(a)
# print("a",a.size())
b = test_label.numpy()
# 将数组保存为 CSV 文件
np.savetxt('FFNNLabel.csv', b)
print("b",b)
loss_test1 = mean_absolute_error(a, b)
loss_test2 = mean_squared_error(a, b)
loss_test3 = np.sqrt(loss_test2)
loss_test4 = nrmse(a, b)
# loss_mean = loss_func(mean, test_label)
time_end = time.time()
time_consumption = time_end - time_start
print('mae:%.2f' % loss_test1, 'mse:%.2f' % loss_test2, 'rmse:%.2f' % loss_test3, 'nrmse: %.2f' % loss_test4,
      '\ntime consumption:%.2f' % time_consumption)

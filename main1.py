# -*- coding: utf-8 -*-
# @Time : 2020-10-10 23:15
# @Author : Kun Wang(Adam Dylan)
# @File : model.py.py
# @Comment : Created By Kun Wang,23:15
# @Completed : Yes
# @Tested : Yes

import time
import os, sys
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import *
from torch.utils.data import DataLoader, random_split
from tqdm import *
from sklearn.model_selection import train_test_split
from dataset import TimeSeriesDataset
from model import DeepAR
import metrics
from sklearn.model_selection import GridSearchCV, KFold

# os.chdir(sys.path[0])
time_start = time.time()


def R2_0(y_test, y_pred):
    SStot = np.sum((y_test - np.mean(y_test)) ** 2)
    SSres = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - SSres / SStot
    return r2


def gaussian_likelihood_loss(z, mu, sigma):
    '''
    Gaussian Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)
    likelihood:
    (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2))
    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2)
    '''
    negative_likelihood = t.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6
    return negative_likelihood.mean()


def Metrics(Pred, Label):
    # 平均绝对误差（Mean Absolute Error，MAE）于评估预测结果和真实数据集的接近程度的程度
    # ，其其值越小说明拟合效果越好
    mae = mean_absolute_error(Label, Pred)
    # 均方误差
    mse = mean_squared_error(Label, Pred)
    rmse = np.sqrt(mse)
    # nrmse
    t1 = np.sum((Pred - Label) ** 2) / np.size(Label)
    t2 = np.sum(abs(Label)) / np.size(Label)
    nrmse = np.sqrt(t1) / t2
    # R
    SStot = np.sum((Label - np.mean(Pred)) ** 2)
    SSres = np.sum((Label - Pred) ** 2)
    r2 = 1 - SSres / SStot
    return np.array([mae, mse, rmse, nrmse, r2])


# 归一化
def Inverse_maxmin(data, min, max):
    seq_len, num_features = data.shape
    for i in range(num_features):
        # data[:,i] 只取第i列的数据
        data[:, i] = (data[:, i] * (max - min) + min)
    return data


import optuna



# 训练参数
params = {
    'target_size': 1,
    'feature_size': 4,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout_rate': 0.2,
    'encode_step': 24,
    'forcast_step': 12,  # 12
    'epochs': 1,  # 10
    'batch_size': 8,
    'lr': 0.001,
    'teacher_prob': 0.8,
    'use_gpu': True,
    # 'fm_k': 84
}

model = DeepAR(params['target_size'],
               params['feature_size'],
               params['hidden_size'],
               params['num_layers'],
               params['dropout_rate'],
               params['forcast_step'],
               params['encode_step'],
               params['teacher_prob'],
               # params['fm_k']
               )

optim = t.optim.AdamW(model.parameters(), lr=params['lr'])  # 优化器
schedule = t.optim.lr_scheduler.CosineAnnealingLR(optim, 5)

scaler = StandardScaler()  # 去均值和方差进行归一化
# 处理实验数据
ds = TimeSeriesDataset(params, scaler)  # 处理协变量和label数据
trainsize = int(0.8 * len(ds))  # 返回ds 0.8长度作为训练数据长度

trainset, testset = random_split(ds, [trainsize, len(ds) - trainsize])  # 划分数据集
# 加载训练集 params['batch_size'] =  每个mini-batch中包含的样本数量
trainLoader = DataLoader(trainset, params['batch_size'], shuffle=False,
                         drop_last=True)  # 加载的数据 一次的batch-size 是否打乱数据 是否去掉末尾数据
testLoader = DataLoader(testset, params['batch_size'], shuffle=False, drop_last=True)

lossfunc = t.nn.MSELoss()

train_loss = []


def train():
    for epoch in range(params['epochs']):
        epoch_loss = 0
        model.train()  # DeepAR的训练模块
        m = 0
        for hisx, hisz, futx, z in tqdm(
                trainLoader):  # 对每个mini-batch执行一次训练 hisx历史观测值的输入特征:24*4 hisz历史时刻的隐状态:24*1 futx未来预测值的输入特征:12*4 z预测值对应的真实标签:12*1
            m += 1
            # print(hisx,hisz)
            optim.zero_grad()  # 将模型的参数梯度初始化为0
            zhat, _ = model.forward(hisx, hisz, futx, z)
            zhat = zhat.reshape(z.shape)
            loss = lossfunc(zhat, z)
            loss.backward()
            optim.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (m + 1)
        train_loss.append(epoch_loss)
        metrics = np.zeros((5,))  # array([0., 0., 0., 0.])
        metrics1 = np.zeros((4,))

        model.eval()
        i = 0
        preds = []
        trues = []
        for hisx, hisz, futx, z in testLoader:  # 测试

            i += 1
            _, zhat = model.forward(hisx, hisz, futx, z)
            # print("预测数据：", zhat.shape)
            z = z[:, -params['forcast_step']:]  # 取出需要预测的label
            zhat = zhat.reshape(z.shape)
            zhat = zhat.detach()
            z = z.detach()
            hisz = hisz.detach()
            zhat = scaler.inverse_transform(zhat.squeeze())  # 反标准化 预测值
            zhat = Inverse_maxmin(zhat, ds.zmin, ds.zmax)
            preds.append(zhat)
            z = scaler.inverse_transform(z.squeeze())
            z = Inverse_maxmin(z, ds.zmin, ds.zmax)
            metrics += Metrics(zhat, z)  # 误差累计

            # 画图
            # if ((i == 38) and (epoch == params['epochs'])):
            if epoch == params['epochs']:  # (i == 38) and (
                a = z.reshape(48)
                b = zhat.reshape(48)
                # plt.figure()
                plt.rcParams['savefig.dpi'] = 300  # 图片像素
                plt.rcParams['figure.dpi'] = 300  # 分辨率
                color = {0: 'red', 1: 'orange', 2: 'green', 3: "yellow", 4: "blue"}
                plt.plot(a, label='True value', c=color[2], alpha=0.3)
                plt.plot(b, label='Predicted value', c=color[2])
                plt.title('The fit of the prediction set')
                plt.legend()
                plt.show()
        preds.append(preds)
        print(params)
        print(
            f'Epoch-{epoch}: MAE={metrics[0] / len(testLoader)};MSE={metrics[1] / len(testLoader)};RMSE={metrics[2] / len(testLoader)};'
            f'NRMSE={metrics[3] / len(testLoader)};'
            f'R={metrics[4] / len(testLoader)}')


train()
time_end = time.time()
print(train_loss)
plt.figure()
plt.plot(train_loss, label='loss')
plt.legend()
plt.show()
print('time consumption:', time_end - time_start)

# -*- coding: utf-8 -*-
# @Time : 2020-10-10 23:15
# @Author : Kun Wang(Adam Dylan)
# @File : model.py
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
import optuna

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

# 训练参数
params = {
    'target_size': 1,
    'feature_size': 4,
    'hidden_size': None,
    'num_layers': None,
    'dropout_rate': None,
    'encode_step': 24,
    'forcast_step': 12,  # 12
    'epochs': 10,  # 10
    'batch_size': 8,
    'lr': None,
    'teacher_prob': None,
    'use_gpu': True,
}

def objective(trial):
    params['hidden_size'] = trial.suggest_int('hidden_size', 32, 256)
    params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
    params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.0, 0.5)
    params['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    params['teacher_prob'] = trial.suggest_float('teacher_prob', 0.6, 1.0)

    model = DeepAR(params['target_size'],
                   params['feature_size'],
                   params['hidden_size'],
                   params['num_layers'],
                   params['dropout_rate'],
                   params['forcast_step'],
                   params['encode_step'],
                   params['teacher_prob'],
                   )

    # 加载数据集
    scaler = StandardScaler()
    dataset = TimeSeriesDataset(params, scaler)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=params['batch_size'],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=params['use_gpu'],
                                  )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=params['batch_size'],
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=params['use_gpu'],
                                 )

    if params['use_gpu'] and t.cuda.is_available():
        device = t.device('cuda')
        model.cuda()
    else:
        device = t.device('cpu')

    optimizer = t.optim.Adam(model.parameters(), lr=params['lr'])

    best_loss = float('inf')
    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0
        for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs, mu, sigma = model(x)

            # 计算损失函数
            loss = gaussian_likelihood_loss(y, mu, sigma)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        # 在测试集上评估模型性能
        model.eval()
        y_true = []
        y_pred = []
        with t.no_grad():
            for x, y in test_dataloader:
                x = x.to(device)
                outputs, mu, sigma = model(x)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(mu.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        # 计算评估指标
        metrics = Metrics(y_pred, y_true)
        mae, mse, rmse, nrmse, r2 = metrics

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_metrics = metrics.copy()
            best_model_state = model.state_dict()

        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {avg_loss:.4f}')
        print(f'Test MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, NRMSE: {nrmse:.4f}, R2: {r2:.4f}')
        print()

    print(f'Best Epoch: {best_epoch}')
    print(f'Best Train Loss: {best_loss:.4f}')
    print(f'Best Test MAE: {best_metrics[0]:.4f}, MSE: {best_metrics[1]:.4f}, RMSE: {best_metrics[2]:.4f}, '
          f'NRMSE: {best_metrics[3]:.4f}, R2: {best_metrics[4]:.4f}')

    return best_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

best_params = study.best_params
print('Best Parameters:', best_params)

# 使用最佳参数进行模型训练
params.update(best_params)
train()

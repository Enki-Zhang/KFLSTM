#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020-07-06 14:35
# @Author : YuHui Li(MerylLynch)
# @File : main.py.py
# @Comment : Created By Liyuhui,14:35
# @Completed : No
# @Tested : No
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as t
from scipy.constants import sigma
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import *
from torch.utils.data import DataLoader, random_split
from tqdm import *

from dataset import TimeSeriesDataset
from model import DeepAR


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
    
    mae = mean_absolute_error(Label, Pred)
    mse = mean_squared_error(Label, Pred)
    rmse = np.sqrt(mse)
    
    
    #nrmse#
    t1 = np.sum((Pred - Label)**2) / np.size(Label)
    t2 = np.sum(abs(Label)) / np.size(Label)
    nrmse = np.sqrt(t1) / t2

    
    return np.array([mae, mse, rmse, nrmse])


params = {
    'target_size': 1,
    'feature_size': 4,
    'hidden_size': 128,
    'num_layers': 1,
    'dropout_rate': 0.2,
    'encode_step': 24,
    'forcast_step': 12,
    'epochs': 10,
    'batch_size': 128,
    'lr': 1e-3,
    'teacher_prob':0.8
}

model = DeepAR(params['target_size'],
               params['feature_size'],
               params['hidden_size'],
               params['num_layers'],
               params['dropout_rate'],
               params['forcast_step'],
               params['encode_step'],
               params['teacher_prob'])

optim = t.optim.AdamW(model.parameters(), lr=params['lr'])
schedule = t.optim.lr_scheduler.CosineAnnealingLR(optim, 5)

scaler = StandardScaler()

ds = TimeSeriesDataset(params, scaler)
trainsize = int(len(ds) * 0.6)
trainset, testset = random_split(ds, [trainsize, len(ds) - trainsize])
trainLoader = DataLoader(trainset, params['batch_size'], shuffle=False, drop_last=True)
testLoader = DataLoader(testset, params['batch_size'], shuffle=False, drop_last=True)

lossfunc = t.nn.MSELoss()


def train():
    for epoch in range(params['epochs']):
        model.train()
        for hisx, hisz, futx, z in tqdm(trainLoader):
            optim.zero_grad()
            zhat, _ = model.forward(hisx, hisz, futx, z)
            zhat = zhat.reshape(z.shape)
            loss = lossfunc(zhat, z)
            loss.backward()
            optim.step()

        metrics = np.zeros((4,))
        model.eval()
        for hisx, hisz, futx, z in testLoader:
            _, zhat = model.forward(hisx, hisz, futx, z)
            z = z[:, -params['forcast_step']:]
            zhat = zhat.reshape(z.shape)
            zhat = zhat.detach()
            z = z.detach()
            hisz = hisz.detach()

            if np.random.rand() < 0.03:
                plt.figure(1, figsize=(12, 4))
                plt.plot(range(0, params['encode_step']), hisz[0, :, 0], color='blue' )
                ymin, ymax = plt.ylim()
                plt.plot(range(params['encode_step'],
                               params['encode_step']
                               + params['forcast_step']), z[0, :, 0]
                         , color='green', label='Real Line')
                plt.plot(range(params['encode_step'],
                               params['encode_step']
                               + params['forcast_step']), zhat[0, :, 0]
                         , color='red', label='Pred Line')
                plt.vlines(params['encode_step'],
                           ymin, ymax, color="blue",
                           linestyles="dashed", linewidth=2)
                plt.ylim(ymin, ymax)
                plt.title(f"On Epochs-{epoch} Testing Plot")
                plt.legend()
                plt.show()
            zhat = scaler.inverse_transform(zhat.squeeze())*14.95+0.05
            z = scaler.inverse_transform(z.squeeze())*14.95+0.05
            metrics += Metrics(zhat, z)
        print(f'Epoch-{epoch}: MAE={metrics[0] / len(testLoader)};MSE={metrics[1] / len(testLoader)};RMSE={metrics[2] / len(testLoader)};NRMSE={metrics[3] / len(testLoader)}')

def loss_func(mu,sigma,zhat):
    return - torch.distributions.Normal(mu,sigma).log_prob(zhat).mean()

def train2():
    for epoch in range(params['epochs']):
        model.train()
        for hisx, hisz, futx, z in tqdm(trainLoader):
            optim.zero_grad()
            _ , _, (mu, simga) = model.forward(hisx, hisz, futx, z)
            loss = loss_func(mu,sigma,z)
            loss.backward()
            optim.step()

        metrics = np.zeros((4,))
        model.eval()
        for hisx, hisz, futx, z in testLoader:
            _, zhat = model.forward(hisx, hisz, futx, z)
            z = z[:, -params['forcast_step']:]
            zhat = zhat.reshape(z.shape)
            zhat = zhat.detach()
            z = z.detach()
            hisz = hisz.detach()

            if np.random.rand() < 0.03:
                plt.figure(1, figsize=(12, 4))
                plt.plot(range(0, params['encode_step']), hisz[0, :, 0], color='blue' )
                ymin, ymax = plt.ylim()
                plt.plot(range(params['encode_step'],
                               params['encode_step']
                               + params['forcast_step']), z[0, :, 0]
                         , color='green', label='Real Line')
                plt.plot(range(params['encode_step'],
                               params['encode_step']
                               + params['forcast_step']), zhat[0, :, 0]
                         , color='red', label='Pred Line')
                plt.vlines(params['encode_step'],
                           ymin, ymax, color="blue",
                           linestyles="dashed", linewidth=2)
                plt.ylim(ymin, ymax)
                plt.title(f"On Epochs-{epoch} Testing Plot")
                plt.legend()
                plt.show()
            zhat = scaler.inverse_transform(zhat.squeeze())*14.95+0.05
            z = scaler.inverse_transform(z.squeeze())*14.95+0.05
            metrics += Metrics(zhat, z)
        print(f'Epoch-{epoch}: MAE={metrics[0] / len(testLoader)};MSE={metrics[1] / len(testLoader)};RMSE={metrics[2] / len(testLoader)};NRMSE={metrics[3] / len(testLoader)}')
        
train()

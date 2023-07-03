#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020-07-12 22:37
# @Author : YuHui Li(MerylLynch)
# @File : model.py
# @Comment : Created By Liyuhui,22:37
# @Completed : No
# @Tested : No

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, random_split
from tqdm import *

from ForcastNet.dataset import TimeSeriesDataset


def Metrics(Pred, Label):
    mae = mean_absolute_error(Label, Pred)
    rmse = np.sqrt(mean_squared_error(Label, Pred))
    mape = np.mean(np.abs((Pred - Label) / (Label + 1e-5))) * 100
    acc = 1 - mape
    return np.array([mae, rmse, mape, acc])


def gaussian_loss(z, mu, sigma):
    """
    Calculate the negative log-likelihood of a given sample Y for a Gaussian with parameters mu and sigma.
    Note that this equation is specific for a single one-dimensional Gaussian mixture component.
    :param z: Target sample
    :param mu: Mean of the Gaussian
    :param sigma: Standard deviation of the Gaussian
    :return log_lik: The computed (negative) log-likelihood.
    """
    n = 1.0
    mu = mu.permute(1, 0, 2)
    sigma = sigma.permute(1, 0, 2)

    loglik = t.mean(n * t.log(sigma) + 0.5 * ((z - mu) ** 2 / sigma ** 2))
    return loglik


class BlockLayer(t.nn.Module):

    def __init__(self, in_feature, out_feature, use_bias=True, n_layer=1):
        super(BlockLayer, self).__init__()
        self.layers = t.nn.ModuleList(
            [
                t.nn.Linear(in_feature, out_feature, use_bias),
                t.nn.BatchNorm1d(out_feature),
                t.nn.ReLU()
            ]
            +
            [
                t.nn.Linear(out_feature, out_feature, use_bias),
                t.nn.BatchNorm1d(out_feature),
                t.nn.ReLU()
            ] * (n_layer - 1)
        )

    def forward(self, input):
        res = input
        for layer in self.layers:
            res = layer(res)
        return res


class BlockCell(t.nn.Module):

    def __init__(self, length, init_dim, in_dim, dim, n_layers=1):
        super(BlockCell, self).__init__()
        self.length = length
        self.init_dim = init_dim
        self.in_dim = in_dim
        self.hidden_dim = dim
        self.layers = t.nn.ModuleList(
            [BlockLayer(self.init_dim, self.hidden_dim, n_layer=n_layers)] +
            [BlockLayer(self.in_dim, self.hidden_dim, n_layer=n_layers)
             for _ in range(length - 1)]
        )

    def __getitem__(self, item):
        return self.layers[item]


class ForcastNet(t.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, hist_len, pred_len, params):
        super(ForcastNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hist_len = hist_len
        self.pred_len = pred_len

        # Input dimension of componed inputs and sequences
        input_dim_comb = input_dim * hist_len

        # Initialise layers
        self.Cells = BlockCell(pred_len, input_dim_comb,
                               input_dim_comb + hidden_dim + output_dim, hidden_dim, n_layers=2)
        self.mu_layer = BlockCell(pred_len, hidden_dim, hidden_dim, output_dim)
        self.sigma_layer = BlockCell(pred_len, hidden_dim, hidden_dim, output_dim)

    def forward(self, input, target):
        bsize = input.shape[0]
        input = input.view(bsize, -1)
        outputs = t.zeros((self.pred_len, bsize, self.output_dim))
        mus = t.zeros((self.pred_len, bsize, self.output_dim))
        sigmas = t.zeros((self.pred_len, bsize, self.output_dim))

        next_input = input
        for i in range(self.pred_len):
            # Step 0: Data Process[Flatten]
            next_input = next_input.view(bsize, -1)

            # Step 1: Cell Compute
            hidden = self.Cells[i](next_input)
            mu = self.mu_layer[i](hidden).relu_()
            sigma = t.nn.functional.softplus(self.sigma_layer[i](hidden)) / 10.

            # Step 2: Result Record
            sample = t.normal(mu, sigma)
            outputs[i] = sample
            mus[i] = mu
            sigmas[i] = sigma
            # Step 3: Prepare for Next
            tinput = None
            if self.training:
                tinput = target[:, i, :]
                next_input = t.cat((input, hidden, tinput), dim=1)
            else:
                tinput = outputs[i, :, :]
                next_input = t.cat((input, hidden, tinput), dim=1)

        return outputs, mus, sigmas


params = {
    'target_size': 1,
    'feature_size': 5,
    'output_size': 1,
    'hidden_size': 32,
    'num_layers': 2,
    'dropout_rate': 0.2,
    'encode_step': 24,
    'forcast_step': 12,
    'epochs': 50,
    'batch_size': 128,
    'lr': 1e-3,
    'teacher_prob': 0.8,
    'scaler': MinMaxScaler()
}

model = ForcastNet(
    params['feature_size'],
    params['hidden_size'],
    params['output_size'],
    params['encode_step'],
    params['forcast_step'],
    params
)

optim = t.optim.AdamW(model.parameters(), lr=params['lr'])
schedule = t.optim.lr_scheduler.CosineAnnealingLR(optim, 5)

ds = TimeSeriesDataset(params, params['scaler'])
trainsize = int(len(ds) * 0.8)
trainset, testset = random_split(ds, [trainsize, len(ds) - trainsize])
trainLoader = DataLoader(trainset, params['batch_size'], shuffle=True, drop_last=True)
testLoader = DataLoader(testset, params['batch_size'], shuffle=True, drop_last=True)


def train():
    for epoch in range(params['epochs']):
        model.train()
        for history, z in tqdm(trainLoader):
            optim.zero_grad()
            outputs, mus, sigmas = model.forward(history, z)
            loss = gaussian_loss(z, mus, sigmas)
            loss.backward()
            optim.step()

        metrics = np.zeros((4,))
        model.eval()
        for history, z in testLoader:
            pred = []
            for i in range(50):
                outputs, mu, sigma = model.forward(history, z)
                pred.append(outputs.detach())

            pred = t.cat(pred, dim=-1)
            p10 = np.quantile(pred, 0.1, axis=-1)
            p50 = np.quantile(pred, 0.5, axis=-1)
            p90 = np.quantile(pred, 0.9, axis=-1)

            if np.random.rand() < 0.03:
                hist = params['scaler'].inverse_transform(history[0, :, 0].view(-1, 1))
                p10s = params['scaler'].inverse_transform(p10[:, 0].reshape(-1, 1))
                p50s = params['scaler'].inverse_transform(p50[:, 0].reshape(-1, 1))
                p90s = params['scaler'].inverse_transform(p90[:, 0].reshape(-1, 1))
                realz = params['scaler'].inverse_transform(z[0, :, 0].reshape(-1, 1))
                plt.figure()
                plt.plot(np.arange(0, params['encode_step']), hist, 'o-', label='History', c='blue')
                plt.plot(np.arange(params['encode_step'], params['encode_step'] + params['forcast_step']),
                         realz,
                         'o-', c='green')
                plt.plot(np.arange(params['encode_step'], params['encode_step'] + params['forcast_step']),
                         p50s[:, 0],
                         '*-', linewidth=0.7, label='mean', c='red')

                plt.fill_between(np.arange(params['encode_step'],
                                           params['encode_step'] + params['forcast_step']),
                                 p10s[:, 0], p90s[:, 0],
                                 color='gray', alpha=0.3, label='Uncertainty')
                plt.legend()
                plt.title(f'On Epoch-{epoch} Test Result')
                plt.show()

            p50 = params['scaler'].inverse_transform(p50.squeeze())
            z = params['scaler'].inverse_transform(z.squeeze())
            metrics += Metrics(p50.transpose(), z)
        print(f'Epoch-{epoch}: MAE={metrics[0] / len(testLoader)};MSE={metrics[1] / len(testLoader)};'
              f'MAPE={metrics[2] / len(testLoader)};Accuracy={metrics[3] / len(testLoader)};')


train()

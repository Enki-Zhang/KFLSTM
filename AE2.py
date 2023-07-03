#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/1/28 11:16
# 数据缺失值填充
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

# 训练参数
params = {
    'target_size': 1,
    'feature_size': 4,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout_rate': 0.2,
    'encode_step': 24,
    'forcast_step': 12,
    'epochs': 10,
    'batch_size': 4,
    'lr': 0.001,
    'teacher_prob': 0.8,
    'fm_k': 84
}


def get_train_data():
    """得到数据
    :return data_x：有缺失值的数据
    :return true_value：缺失数据的原始真实值
    :return data_y：原问题中待预测的label
    """

    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:  # 函数后面跟着的箭头是函数返回值的类型建议符，用来说明该函数返回的值是什么类型
        return torch.tensor(data=dataframe_series.values)

    import copy
    from sklearn.datasets import make_classification
    # 取出原始数据
    data_x = pd.read_csv('HMW_ALL.csv')
    data_x = pd.DataFrame(data_x)
    # scaler = StandardScaler()  # 去均值和方差进行归一化
    # 处理实验数据
    # ds = TimeSeriesDataset(params, scaler)  # 处理协变量和label数据
    # trainsize = int(0.8 * len(ds))  # 返回ds 0.8长度作为训练数据长度
    #
    # trainset, testset = random_split(ds, [trainsize, len(ds) - trainsize])  # 划分数据集
    # 数据读取
    # trainLoader = DataLoader(trainset, params['batch_size'], shuffle=False,
    #                          drop_last=True)  # 加载的数据 一次的batch-size 是否打乱数据 是否去掉末尾数据
    # testLoader = DataLoader(testset, params['batch_size'], shuffle=False, drop_last=True)

    # data_x, data_y = make_classification(n_samples=1000, n_classes=4, n_features=40, n_informative=4,
    #                                      random_state=0)  # 6个特征
    # data_x = pd.DataFrame(data_x)
    # data_x.columns = ["x{}".format(i + 1) for i in range(39)] + ["miss_line"]  # 取列名
    # true_data = copy.deepcopy(data_x)  # 深复制
    # 在miss_line这一列删除20%的数据,来模拟缺失值的场景
    # drop_index = data_x.sample(frac=0.1).index  # 有缺失值的index 随机抽取0.1的数据
    # data_x.loc[drop_index, "miss_line"] = np.nan
    # true_value = true_data.loc[drop_index, 'miss_line']  # 空值的真实值
    # 开始构造数据
    # data_x为全部的数据（包含完整数据、有缺失项的数据）
    # full_x = data_x.drop(drop_index)
    # lack_x = data_x.loc[drop_index]
    # return get_tensor_from_pd(data_x).float(), get_tensor_from_pd(lack_x).float(), true_value
    return get_tensor_from_pd(data_x).float()


class AutoEncoder(nn.Module):
    def __init__(self, input_size=300, hidden_layer_size=20):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        # 输入与输出的维度相同
        self.input_size = input_size
        self.output_size = input_size

        self.encode_linear = nn.Linear(self.input_size, hidden_layer_size)
        self.decode_linear = nn.Linear(hidden_layer_size, self.output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        # encode
        encode_linear = self.encode_linear(input_x)
        encode_out = self.relu(encode_linear)
        # decode
        decode_linear = self.decode_linear(encode_out)  # =self.linear(lstm_out[:, -1, :])
        predictions = self.sigmoid(decode_linear)
        return predictions


if __name__ == '__main__':
    # 得到数据
    print("开始训练Auto Encoder")
    full_data = get_train_data()
    # true_x = full_data
    # lack_data = full_data
    # full_data = get_train_data()
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(full_data),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=20,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=4,  # 多进程（multiprocess）来读数据
    )
    # 建模三件套：loss，优化，epochs
    model = AutoEncoder(input_size=full_data.size()[1])  # 模型
    loss_function = nn.MSELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 100
    # 开始训练
    model.train()
    for i in range(epochs):
        epoch_loss: list = []
        for seq in train_loader:
            # seq = torch.where(torch.isnan(seq), torch.full_like(seq, 0), seq)  # 全0填充缺失值
            seq = seq[0]
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            single_loss = loss_function(y_pred, seq)
            single_loss.backward()
            optimizer.step()
            epoch_loss.append(float(single_loss.detach()))
        print("EPOCH", i, "LOSS: ", np.mean(epoch_loss))
    # 开始填充缺失值
    lack_loader = Data.DataLoader(
        dataset=Data.TensorDataset(lack_data),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=20,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=4,  # 多进程（multiprocess）来读数据
    )
    model.eval()
    pred_lack = np.array([])
    for seq in lack_loader:
        seq = seq[0]
        # 每个seq[:,-1]都是缺失值的位置
        seq = torch.where(torch.isnan(seq), torch.full_like(seq, 0), seq)  # 全0填充缺失值
        y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
        lack_pred = y_pred[:, -1]  # AutoEncoder预测的缺失值
        pred_lack = np.append(pred_lack, np.array(lack_pred.detach().numpy()))
    np.savetxt('./HMW_All_save.', pred_lack, delimiter=",")
    print("预测结果的MSE：", mean_squared_error(full_data, pred_lack))

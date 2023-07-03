# -*- coding: utf-8 -*-
# @Time : 2020-10-10 23:15
# @Author : Kun Wang(Adam Dylan)
# @File : model.py.py
# @Comment : Created By Kun Wang,23:15
# @Completed : Yes
# @Tested : Yes

import numpy as np
from sklearn.preprocessing import *
from torch.utils.data import Dataset
import torch as t


class TimeSeriesDataset(Dataset):

    def minmaxscale(self, data: np.ndarray):  # 将每一列即将每一维特征线性地映射到指定的区间，通常是[0, 1]。
        seq_len, num_features = data.shape  # 协变量的大小
        for i in range(num_features):
            min = data[:, i].min()  # 取出第i列的所有数据最小的
            max = data[:, i].max()
            data[:, i] = (data[:, i] - min) / (max - min)
        return data, min, max

    def __init__(self, params, scaler=StandardScaler()):
        import pandas as pd

        self.encode_step = params['encode_step']
        self.forcast_step = params['forcast_step']
        # rawdata = pd.read_csv('dataset2.csv')
        # rawdata = pd.read_csv('平滑1000oxy.csv')
        rawdata = pd.read_csv('./KRMData/oxy/原始数据平滑训练和验证集不平滑测试集.csv')
        # rawdata = pd.read_csv('./KRMData/oxy/海门湾label=4.csv')
        # rawdata = pd.read_csv('去掉时间原始数据平滑训练和验证集不平滑测试集oxy.csv')
        # rawdata = pd.read_csv('海门湾2000x5.csv')

        # 1.带0预测（label补足）
        self.features = rawdata.fillna(0)  # 用0值填充缺失值  label在最后一列
        self.features = self.features.to_numpy().astype('float32')  # 转为数组
        self.features = self.features[:, 0:4]  # 取出前四列 协变量
        self.features, self.xmin, self.xmax = self.minmaxscale(self.features)  # 协变量的数据映射

        self.label = rawdata.fillna(method='ffill', inplace=False)  # label缺失数值处理 用前一个非缺失值去填充该缺失值并创建一个副本
        self.label = self.label.to_numpy().astype('float32')
        self.label = self.label[:, 4]  # 取出最后一列作为label
        self.label, self.zmin, self.zmax = self.minmaxscale(self.label.reshape(-1, 1))

        # 2.不带0预测（全数据补足）
        # target = rawdata.fillna(method = 'ffill', inplace=False)   
        # target = target.to_numpy().astype('float32')
        # target = self.minmaxscale(target)

        # features = rawdata.fillna(method = 'ffill', inplace=False)  
        # features = features.to_numpy().astype('float32')
        # features = self.minmaxscale(features)

        # self.featuress = rawdata[:, 0:4]
        # self.label = testdata[:, 4]

        self.features, _, _ = self.minmaxscale(self.features)  # 再做一次协变量的数据映射
        self.scaler = scaler  # 对每一个特征维度去均值和方差归一化
        self.scaler.fit(self.label.reshape(-1, 1))  # 预处理label数据 reshape(-1,1)将行转为1列
        self.label = scaler.transform(self.label.reshape(-1, 1)).astype('float32')  # label数据归一化
    #  __getitem__用于获取指定索引的元素
    def __getitem__(self, index):
        # features
        # [index + 1, t0] [t0 + 1, T]
        # lag input
        # [index, t0 - 1]

        # rowfea = self.featuress[index]

        # Step 1: lagged size adjust
        index += 1

        # Step 2: history featuress
        start = index
        end = start + self.encode_step
        hisx = self.features[start:end]

        # Step 3: history inputs
        hisz = self.label[start - 1:end - 1]

        # Step 4: future featuress
        start = end + 1
        end = start + self.forcast_step
        futx = self.features[start:end]

        # Step 5: targets
        z = self.label[index: index + self.encode_step + self.forcast_step]

        # ha = np.mean(self.label[0:index + self.encode_step])
        # HA = []
        # for i in range(12):
        #         HA.append(np.float32(ha))
        # HA = np.array(HA)
        return hisx, hisz, futx, z  # , HA

    def __len__(self):
        return len(self.features) - self.forcast_step - self.encode_step - 1  # 整个数据集的长度

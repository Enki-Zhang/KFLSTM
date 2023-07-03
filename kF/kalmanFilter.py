#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/1/30 15:45
import numpy as np
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import *
import matplotlib.pyplot as plt
from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother
import pandas as pd

# 0值替换为nan

File = "海门湾label=4.csv"  # 未平滑数据

df = pd.read_csv(File)
# df.dropna(inplace=True)
# df[df == 0] = np.nan  # df
# print(df)
# np.savetxt(Path+"训练和验证集.csv", df, fmt='%.01f', delimiter=",")
#
num_train = int(len(df) * 0.8)  # 划分测试集
# num_test = int(len(df) * 0.2)  # 划分验证集
# num_vali = len(df) - num_train - num_test
train_df = df[0:num_train]
test_df = df[num_train:]
# test_df[test_df == np.nan] = 0
test_df = test_df.values
test_df[np.isnan(test_df)] = 0
# print("test", type(test_df))
# 平滑验证和训练
smoother = KalmanSmoother(component='level_trend',
                          component_noise={'level': 0.5, 'trend': 0.5})

# tem,PH,elec,tur,oxy
smoother.smooth(train_df[['tem', 'PH', 'elec', 'tur', 'oxy']].T)

# np.savetxt(Path+"原始数据平滑训练集.csv", smoother.smooth_data.T, fmt='%.01f', delimiter=",")

# print(df.shape,type(smoother.smooth_data.T))
newData = np.vstack((smoother.smooth_data.T, test_df))
np.savetxt("原始数据平滑训练集不平滑测试集参数实验.csv", newData, fmt='%.2f', delimiter=",")

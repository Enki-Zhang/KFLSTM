# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:01:42 2020

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def minmaxscale(data: np.ndarray):
    seq_len, num_features = data.shape
    for i in range(num_features):
        min = data[:, i].min()
        max = data[:, i].max()
        data[:, i] = (data[:, i] - min) / (max - min)
    return data, min, max


data = pd.read_csv('海门湾1000x5.csv', encoding='unicode_escape')
data = data.fillna(0)
data = np.array(data)
plt.figure(1)
data, _, _ = minmaxscale(data)
# data, _, _ = StandardScaler()
plt.figure(1)
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 13,
# }
plt.xlabel('Time Step')
plt.scatter(np.array(range(100)), data[0:100, 1].reshape(100), color='#EC7063', label='PH')
plt.scatter(np.array(range(100)), data[0:100, 0].reshape(100), color='#3498DB', label='Temperature')
plt.legend()
plt.show()

plt.figure(2)
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# font2 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 13,
# }
plt.xlabel('Time Step')
plt.scatter(np.array(range(100)), data[0:100, 2].reshape(100), color='#1ABC9C', label='Conductivity')
plt.scatter(np.array(range(100)), data[0:100, 3].reshape(100), color='#5D6D7E', label='Turbility')
plt.legend()
plt.show()
# plt.figure(2)
# plt.subplot(2,1,1)
# plt.plot(data[0:99,2])
# plt.subplot(2,1,2)
# plt.plot(data[0:99,3])

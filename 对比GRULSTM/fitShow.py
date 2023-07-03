#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/3/31 10:33
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

GRUPRE = pd.read_csv("GRUPRE.csv")
LSTMPre = pd.read_csv("LSTMPre.csv")
LSTMTrue = pd.read_csv("LSTMTrue.csv")

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
color = {0: '#f1948a', 1: 'orange', 2: 'black', 3: "yellow", 4: "blue"}
plt.plot(LSTMTrue, label='True value', c=color[2], )
plt.plot(LSTMPre, label='Predicted of KF-LSTM', c=color[4])
plt.plot(GRUPRE, label='Predicted of kF-GRU', c=color[0], )
plt.title('Fitting performance of LSTM and GRU models')
plt.legend()
plt.show()

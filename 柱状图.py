# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:10:38 2020

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

n_groups = 8

MAE = (4.36, 1.88, 1.85, 1.2, 2.28, 1.73, 1.75, 0.57)
# std_men = (2, 3, 4, 1, 2)

MSE = (21.4, 6.29, 4.58, 2.26, 6.52, 3.85, 3.91, 0.64)
# std_women = (3, 5, 2, 3, 3)

RMSE = (4.62, 2.51, 2.14, 1.50, 2.55, 1.96, 1.98, 0.8)

NRMSE = (0.97, 2.62, 0.66, 0.39, 0.79, 0.48, 0.50, 0.16)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index - bar_width, MAE, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='MAE')

rects2 = ax.bar(index, MSE, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='MSE')

rects3 = ax.bar(index + bar_width, RMSE, bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='RMSE')

rects4 = ax.bar(index + 2 * bar_width, NRMSE, bar_width,
                alpha=opacity, color='#566573',
                error_kw=error_config,
                label='NRMSE')

ax.set_xlabel('Methods')
ax.set_title('The performances of the 8 methods ')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('HA', 'Arima', 'LR', 'XGBoost', 'FFNN', 'FC-LSTM', 'FC-GRU', 'FM-GRU'))
ax.legend()

fig.tight_layout()
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.show()

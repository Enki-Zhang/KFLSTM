# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:10:18 2020

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mae = np.array(pd.read_csv('k的取值的影响MAE.csv'))
mse = np.array(pd.read_csv('k的取值的影响MSE.csv'))
rmse = np.array(pd.read_csv('k的取值的影响RMSE.csv'))
nrmse = np.array(pd.read_csv('k的取值的影响NRMSE.csv'))

plt.figure(figsize=(8, 6), dpi=200)

plt.subplot(2, 2, 1)
x = mae[:, 0]
y = mae[:, 1]
plt.title('The impact of K for MAE')
plt.xlabel('K')

plt.plot(x, y, color='#EC7063', label='MAE')
plt.legend(loc='lower left')

plt.subplot(2, 2, 2)
x = mse[:, 0]
y = mse[:, 1]
plt.title('The impact of K for MSE')
plt.xlabel('K')

plt.plot(x, y, color='#48C9B0', label='MSE')
plt.legend(loc='lower left')

plt.subplot(2, 2, 3)
x = rmse[:, 0]
y = rmse[:, 1]
plt.xlabel('K')
plt.title('The impact of K for RMSE')

plt.plot(x, y, color='#5DADE2', label='RMSE')
plt.legend(loc='lower left')

plt.subplot(2, 2, 4)
x = nrmse[:, 0]
y = nrmse[:, 1]
plt.xlabel('K')
plt.title('The impact of K for NRMSE')

plt.plot(x, y, color='#707B7C', label='NRMSE')
plt.legend(loc='lower left')

plt.tight_layout()

plt.show()

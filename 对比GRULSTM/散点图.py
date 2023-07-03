#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/5/13 20:53
import matplotlib.pyplot as plt
import pandas as pd

# np.savetxt('True.csv', a)
# np.savetxt('KFLSTMPRE.csv', b)
# np.savetxt('KFGRUPRE1.csv', c)
# np.savetxt('GRUPRE1.csv', d)
# np.savetxt('LSTMPRE1.csv', e)
# 提取观测值和预测值列
observed = pd.read_csv('True.csv')
predictedLSTM = pd.read_csv('LSTMPRE1.csv')
predictedKFLSTM = pd.read_csv('KFLSTMPRE.csv')
predictedKFGRU = pd.read_csv('KFGRUPRE1.csv')
predictedGRU = pd.read_csv('GRUPRE1.csv')
predictedXGB = pd.read_csv('xgb.csv')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图
plt.scatter(observed, predictedGRU, label='GRU-seq2seq',marker='^', color='#fe8600')
plt.scatter(observed, predictedLSTM, label='LSTM-seq2seq',marker='o',color='blue')
plt.scatter(observed, predictedKFLSTM, label='KF-LSTM',marker='s',color='green')
plt.scatter(observed, predictedKFGRU, label='KF-GRU',marker='d',color='purple')
# plt.scatter(observed, predictedXGB, label='XGboost',marker='d',color='#018101')

# 添加图例
plt.legend()
# 绘制x=y的射线
x = y = range(0, 12)
plt.plot(x, y, color='red', linewidth=1)
plt.text(9, 10, 'Y=X',color='red',fontsize=20,fontfamily='serif', fontstyle='italic')
plt.xlabel('Observed Values(DO mg/l)')
plt.ylabel('Predicted Values(DO mg/l)')
# plt.title('Observed vs. Predicted Values')
plt.tight_layout()  # 去掉图形两边空白
plt.show()


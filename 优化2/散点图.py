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
observed = pd.read_csv('TRUE10003.csv')
predictedLSTM = pd.read_csv('LSTMPRE110003.csv')
predictedKFLSTM = pd.read_csv('KFLSTMPRE10003.csv')
predictedKFLSTM = pd.read_csv('KFLSTMPRE10003.csv')
predictedKFGRU = pd.read_csv('KFGRUPRE110003.csv')
predictedGRU = pd.read_csv('GRUPRE110003.csv')
# predictedXGB = pd.read_csv('xgb.csv')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图
# plt.scatter(observed, predictedGRU, label='GRU-seq2seq',marker='^', color='#fe8600')
# plt.scatter(observed, predictedLSTM, label='LSTM',marker='o',color='#fe8600')
# plt.scatter(observed, predictedKFGRU, label='KF-GRU',marker='d',color='purple')
plt.scatter(observed[:1800], predictedKFLSTM[:1800], label='KF-LSTM',marker='^',color='red')

plt.scatter(observed[:1000], predictedKFLSTM[:1000], label='KF-LSTM',marker='o',color='blue')

# plt.scatter(observed, predictedKFLSTM, label='KF-LSTM',marker='o',color='#fe8600')
plt.text(0.85, 0.08, fr'$R^2={0.87:.2f}$',
         ha='center', va='center', transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style':'oblique','size': 20})


# 添加图例
plt.legend()
# 绘制x=y的射线
x = y = range(0, 12)
plt.plot(x, y, color='green', linewidth=1)
plt.text(9, 10, 'Y=X',color='red',fontsize=20,fontfamily='serif', fontstyle='italic')
plt.xlabel('Observed Values(DO)')
plt.ylabel('Predicted Values(DO)')
# plt.title('Observed vs. Predicted Values')
plt.tight_layout()  # 去掉图形两边空白
plt.show()


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
# predictedLSTM = pd.read_csv('LSTMPRE11000.csv')
predictedKFLSTM = pd.read_csv('KFLSTMPRE10003.csv')
# predictedKFGRU = pd.read_csv('KFGRUPRE11000.csv')
# predictedGRU = pd.read_csv('GRUPRE11000.csv')
# predictedXGB = pd.read_csv('xgb.csv')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图
# plt.scatter(observed, predictedGRU, label='GRU-seq2seq',marker='^', color='#fe8600')
# plt.scatter(observed, predictedLSTM, label='LSTM-seq2seq',marker='o',color='blue')
plt.scatter(observed, predictedKFLSTM, label='Test set',marker='o',color='blue')
# plt.scatter(observed, predictedKFGRU, label='KF-GRU',marker='d',color='purple')
# 添加 R2 标签
plt.text(0.85, 0.18, fr'$R^2={0.93:.2f}$', ha='center', va='center', transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style':'oblique','size': 20})
lr_label = plt.text(11.42, -0.35,
                    f'(f)KF-LSTM', ha='right', va='bottom',fontsize=20 )
# 设置LR标签背景颜色为灰色
lr_label.set_bbox(dict(facecolor='grey', alpha=0.2, edgecolor='none'))
# 添加图例

# 添加图例
plt.legend()
# 绘制x=y的射线
x = y = range(0, 12)
plt.plot(x, y, color='red', linewidth=1)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.text(9, 10, 'Y=X',color='red',fontsize=20,fontfamily='serif', fontstyle='italic')
plt.xlabel('Observed Values(DO mg/l)',fontsize=20)
plt.ylabel('Predicted Values(DO mg/l)',fontsize=20)
# plt.title('Observed vs. Predicted Values')
plt.tight_layout()  # 去掉图形两边空白
plt.show()


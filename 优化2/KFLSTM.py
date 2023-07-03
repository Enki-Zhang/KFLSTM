#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/5/13 20:53
import matplotlib.pyplot as plt
import pandas as pd
# 提取观测值和预测值列
observed = pd.read_csv('TRUE10003.csv')
predictedLSTM = pd.read_csv('LSTMPRE110003.csv')
KFLSTMTrain = pd.read_csv('KFLSTMPRE10003.csv')
KFLSTMTest = pd.read_csv('KFLSTMPRE10004.csv')
predictedKFGRU = pd.read_csv('KFGRUPRE110003.csv')
predictedGRU = pd.read_csv('GRUPRE110003.csv')
# predictedXGB = pd.read_csv('xgb.csv')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图
# plt.scatter(observed, predictedGRU, label='GRU-seq2seq',marker='^', color='#fe8600')
# plt.scatter(observed, predictedLSTM, label='LSTM',marker='o',color='#fe8600')
# plt.scatter(observed, predictedKFGRU, label='KF-GRU',marker='d',color='purple')
plt.scatter(observed[1300:2300], KFLSTMTrain[1300:2300], label='Train set',marker='^',color='red')

plt.scatter(observed[600:1400], KFLSTMTrain[600:1400], label='Test set',marker='o',color='blue')

# plt.scatter(observed, predictedKFLSTM, label='KF-LSTM',marker='o',color='#fe8600')
plt.text(0.75, 0.28, fr'$R^2_{{\mathrm{{train}}}}={0.95:.2f}$',ha='center', va='center',
         transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style':'oblique','size': 20})
plt.text(0.75, 0.18, fr'$R^2_{{\mathrm{{test}}}}={0.94:.2f}$',ha='center', va='center',
         transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style':'oblique','size': 20})

lr_label = plt.text(11.42, -0.35,
                    f'(g)KF-LSTM', ha='right', va='bottom',fontsize=20 )
# 设置LR标签背景颜色为灰色
lr_label.set_bbox(dict(facecolor='grey', alpha=0.2, edgecolor='none'))
# 添加图例
# 添加图例
plt.legend()
# 绘制x=y的射线
x = y = range(0, 12)
plt.plot(x, y, color='green', linewidth=1)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.text(9, 10, 'Y=X',color='green',fontsize=20,fontfamily='serif', fontstyle='italic')
plt.xlabel('Observed Values(DO mg/l)',fontsize=20)
plt.ylabel('Predicted Values(DO mg/l)',fontsize=20)
# plt.title('Observed vs. Predicted Values')
plt.tight_layout()  # 去掉图形两边空白
plt.show()


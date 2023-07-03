#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/22 23:23
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/22 14:54
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/22 14:28
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/5/13 20:53
import matplotlib.pyplot as plt
import pandas as pd

# 提取观测值和预测值列
observed = pd.read_csv('XGBLabel.csv')
predicted = pd.read_csv('XGBPre.csv')
# train = pd.read_csv('MLPTrainPre.csv')
# predictedKFLSTM = pd.read_csv('KFLSTMPRE.csv')
# predictedKFGRU = pd.read_csv('KFGRUPRE1.csv')
# predictedGRU = pd.read_csv('GRUPRE1.csv')
# predictedXGB = pd.read_csv('xgb.csv')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图
# plt.scatter(observed, train, label='train',marker='o', color='red')
plt.scatter(observed, predicted, label='Test set', marker='o', color='blue')
# plt.scatter(observed, predictedLSTM, label='LSTM-seq2seq',marker='o',color='blue')
# plt.scatter(observed, predictedKFLSTM, label='KF-LSTM',marker='s',color='green')
# plt.scatter(observed, predictedKFGRU, label='KF-GRU',marker='d',color='purple')
# plt.scatter(observed, predictedXGB, label='XGboost',marker='d',color='#018101')
plt.text(0.85, 0.18, fr'$R^2={0.78:.2f}$', ha='center', va='center', transform=plt.gca().transAxes,
         fontdict={'family': 'serif', 'style': 'oblique', 'size': 20})
# 添加标签
lr_label = plt.text(11.42, -0.35,
                    f'(d)XGBoost', ha='right', va='bottom',fontsize=20 )
# 设置LR标签背景颜色为灰色

lr_label.set_bbox(dict(facecolor='grey', alpha=0.2, edgecolor='none'))
# 添加图例
plt.legend(loc='upper left')
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

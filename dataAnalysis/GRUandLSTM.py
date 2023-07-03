#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/2/27 16:18


# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

name_list = ['MAE', 'MSE', 'RMSE', 'RSE']
# KFLSTM = [0.5040432565013532, 0.46737286522377786, 0.6515326551541283, 0.8541937536228273]
# LSTM = [0.5564997427324455, 0.593098782805732, 0.7503122637379425, 0.8948936757026489]
# GRU = [0.8993558427553845, 1.5125253863375416, 1.166098682341837, 0.6938534797410305]
# x = list(range(len(KFLSTM)))
# total_width, n = 0.8, 2
# width = total_width / n
# color = {0: 'red', 1: 'orange', 2: 'green', 3: "yellow", 4: "blue"}
# plt.bar(x, KFLSTM, width=width, label='KF-LSTM', fc=color[3])
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, GRU, width=width, label='GRU', tick_label=name_list, fc=color[0])
# plt.bar(x, LSTM, width=width, label='LSTM', tick_label=name_list, fc=color[4])
# plt.legend()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
barWidth = 0.1
#  MAE=0.5903881905831031;MSE=0.6395922365823574;RMSE=0.7809802630848;NRMSE=0.15344789589502184;R=0.8704455016080374  KFGRU
# MAE=0.6305882788885868;MSE=0.751924383099329;RMSE=0.8405705219394625;NRMSE=0.16966842857140607;R=0.8707134506108881  GRU raw
# 设置柱子的高度
bars1 = [0.5040432565013532, 0.46737286522377786, 0.6515326551541283, 0.8541937536228273]  # KF-LSTM
bars2 = [0.5564997427324455, 0.593098782805732, 0.7503122637379425, 0.8948936757026489]  # LSTM
bars3 = [0.5903881905831031, 0.6395922365823574, 0.7809802630848, 0.8704455016080374]  # KF-GRU
bars4 = [0.6305882788885868, 0.751924383099329, 0.8405705219394625, 0.8707134506108881]  # GRU

r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
color = {0: 'red', 1: 'orange', 2: 'green', 3: "yellow", 4: "blue"}
# 创建柱子
plt.bar(r1, bars1, color='#dae3f3', width=barWidth, edgecolor='#dae3f3', tick_label=name_list, label='KF-LSTM')
plt.bar(r2, bars2, color=color[4], width=barWidth, edgecolor=color[4], tick_label=name_list, label='LSTM-se2seq')
plt.bar(r3, bars3, color=color[2], width=barWidth, edgecolor=color[2], tick_label=name_list, label='KF-GRU')
plt.bar(r4, bars4, color='#06bd59', width=barWidth, edgecolor='#06bd59', tick_label=name_list, label='GRU-seq2seq')
# 添加x轴名称
plt.xticks([r + barWidth+0.05 for r in range(len(bars1))], name_list)
# 创建图例
plt.tight_layout()  # 去掉图形两边空白
plt.legend()
# 展示图片
plt.show()

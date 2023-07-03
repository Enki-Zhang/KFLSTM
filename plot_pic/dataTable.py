#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/2/19 11:05
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 展示中文字体
plt.rcParams["axes.unicode_minus"] = False  # 处理负刻度值
# 列名
col = ['Temperature', 'PH', 'electrical conductivity', 'Turbidity', 'Dissolved oxygen', ]
# 行名
row = ['count', 'mean', 'Missing value number', 'Minimum value', 'Maximum value']
# 表格数据
cellTexts = [[13128, 24.655, 371, 0, 35], [13128, 7.4012, 376, 0, 9.86], [13128, 4596.68, 365, 0, 46878],
             [13128, 55.19, 364, 0, 500], [13128, 5.2366, 418, 0, 15]]
plt.figure(dpi=300)
plt.table(cellText=cellTexts,  # 简单理解为表示表格里的数据
          # colWidths=[0.5] * 6,  # 每个小格子的宽度 * 个数，要对应相应个数
          colLabels=row,  # 每列的名称
          rowLabels=col,  # 每行的名称（从列名称的下一行开始）
          cellLoc="center",  # 行名称的对齐方式
          rowLoc='center',  # 表格所在位置
          loc="center"
          )
# plt.scale(1,2)
# plt.axis('tight')
plt.axis('off')
plt.show()

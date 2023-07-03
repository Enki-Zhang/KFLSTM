#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/24 14:06
import pandas as pd
# df = pd.read_csv("hmw.csv")
df = pd.read_csv("平滑1000oxy.csv")
df = df.dropna()

correlation = df.corr(method='pearson')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率


sns.heatmap(correlation, annot=False, cmap='RdBu')
# 设置标签字体大小为12
# 设置字体大小为12
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.rcParams.update({'font.size': 10})
plt.ylim(0,len(correlation))
plt.tight_layout()  # 去掉图形两边空白
plt.show()


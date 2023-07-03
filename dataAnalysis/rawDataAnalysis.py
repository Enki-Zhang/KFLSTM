#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/1/30 11:13
# 分析数据

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import MultipleLocator

Path = "../KRMData/oxy/"
# FileName = "海门湾label=4.csv"  # 未平滑数据
FileName = "原始数据平滑训练和验证集不平滑测试集.csv"  # 平滑数据
Raw = "海门湾label=4.csv"  # 原始数据
File = Path + FileName  # 平滑数据
rawFile = Path + Raw  # 原始数据
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
print("读取数据")
data = pd.read_csv(File, usecols=[0, 1, 2, 3, 4])
rawData = pd.read_csv(rawFile, usecols=[0, 1, 2, 3, 4])
# newData = pd.read_csv(Path + "原始数据平滑训练集不平滑测试集.csv", usecols=[0, 1, 2, 3, 4])
WData = np.array(data)
RData = np.array(rawData)
# NData = np.array(newData)
# length = np.shape(WData)[0] * 0.8
# print("len", length)
# print("WData.shape", WData.shape, type(WData))
# print(data.columns)
# print(data.index)
# print(type(data))
# 数据对比
# 断层数据显示


#
for i, name in enumerate(data.columns):  # Index(['time', 'tem', 'PH', 'oxy', 'elec', 'tur'], dtype='object')
    color = {0: 'red', 1: 'orange', 2: 'green', 3: "yellow", 4: "blue"}
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.figure(figsize=(24, 4))
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700], [8345, 8445, 8545, 8645, 8745, 8845, 8945],fontsize=14)
    plt.plot(WData[8345:8954, i], c=color[4], label=name, )  # 平滑数据
    plt.plot(RData[8345:8954, i], c=color[4], label=name,alpha=0.3)  # 原始数据
    if name == 'tem':
        plt.ylabel('℃',fontsize=20)
    if name == 'PH':
        plt.ylabel('ab.unit',fontsize=20)
    if name == 'oxy':
        plt.ylabel('mg/l',fontsize=20)
    if name == 'elec':
        plt.ylabel('μ/cm',fontsize=20)
    if name == 'tur':
        plt.ylabel('NTU',fontsize=20)
    plt.xlabel('Data Number',fontsize=20)
    # 设置y轴刻度标签的字体大小为14
    plt.tick_params(axis='y', labelsize=14)
    plt.tight_layout()  # 去掉图形两边空白
    plt.legend()
    # plt.savefig("./pic/litPic" + "raw" + name + ".png")
    plt.show()
    print(i, name)

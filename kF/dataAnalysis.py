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
FileName = "海门湾label=4.csv"  # 未平滑数据
File = Path + FileName
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
print("读取数据")
# data = pd.read_csv(File, usecols=[0, 1, 2, 3, 4])
data = pd.read_csv(File)
newData = pd.read_csv("原始数据平滑训练集不平滑测试集参数实验.csv")
# 添加列名
# col_names = ["tem", "PH", "elec", "tur", "oxy"]
# newData = np.vstack((col_names, newData))
WData = np.array(data)
NData = np.array(newData)
length = np.shape(WData)[0] * 0.8
print("len", length)
print("WData.shape", WData.shape, type(WData))
print(data.columns)
print(data.index)

# 数据对比
color = {0: 'red', 1: 'orange', 2: 'green', 3: "yellow", 4: "blue"}
# 拟合需要在平滑数据后加上列名
for i, name in enumerate(data.columns):  # Index(['time', 'tem', 'PH', 'oxy', 'elec', 'tur'], dtype='object')
    # if name == 'oxy':
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.figure(figsize=(24, 4))
    # 全部数据
    # plt.plot(data.index, WData[:, i], c=color[2], label=name, alpha=0.3)  # 原始数据
    # ymin, ymax = plt.ylim()
    # # plt.vlines(length, ymin,ymax,color="blue",
    # #            linestyles="dashed", linewidth=2)
    #
    # plt.axvline(length, color='blue', linestyle='dashed')
    # plt.plot(data.index, NData[:, i], c=color[2], label=name + ' smooth')
    # 部分数据
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700], [8345, 8445, 8545, 8645, 8745, 8845, 8945])
    plt.plot(NData[8345:8954, i], c=color[4], label=name + ' smooth')
    plt.plot(WData[8345:8954, i], c=color[0], label=name, alpha=0.7)  # 原始数据4

    if name == 'tem':
        plt.ylabel(r"$\degree$C", fontsize=20)
    if name == 'PH':
        plt.ylabel('ab.unit', fontsize=20)
    if name == 'oxy':
        plt.ylabel('mg/l', fontsize=20)
    if name == 'elec':
        plt.ylabel('μ/cm', fontsize=20)
    if name == 'tur':
        plt.ylabel('NTU', fontsize=20)
    plt.xlabel('Data Number', fontsize=20)
    # 设置y轴刻度标签的字体大小为14
    plt.tick_params(axis='y', labelsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.tight_layout()  # 去掉图形两边空白
    plt.legend(fontsize=20)
    plt.tight_layout()  # 去掉图形两边空白
    plt.savefig("dataAndSmooth_0.5" + name + ".png")
    plt.show()
    print(i, name)

# xdata = WData[:, 1]
# ydata = NData[:, 1]
# print(xdata[0])
# print("数据大小", xdata.size)
# xdata = data.ix[:, 'oxy']  # 将csv中列名为“列名1”的列存入xdata数组中
# # 如果ix报错请将其改为loc
# xdata = [0:1200, 0]  # num处写要读取的列序号，0为csv文件第一列
# ydata = newData[1].iloc[0:1200, 0]

# plt.figure(figsize=(24, 4))  # 设置长宽比
# plt.plot(data.index, WData[:, 0], c=color[2], label="PH", alpha=0.3)  # 原始数据
# plt.plot(data.index, NData[:, 0], c=color[2], label="PH" + ' smooth')
# plt.plot(xdata, 'bo-', label=u'', linewidth=1)
# plt.plot(ydata, 'bo-', label=u'', linewidth=1, color="red")
# plt.title(u"oxy", size=10)  # 设置表名为“表名”
# plt.legend()
# # plt.xlabel(u'x轴名', size=10)  # 设置x轴名为“x轴名”
# # plt.ylabel(u'y轴名', size=10)  # 设置y轴名为“y轴名”
# plt.show()

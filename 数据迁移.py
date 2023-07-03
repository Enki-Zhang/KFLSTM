# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:37:19 2020

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

row = pd.read_csv('海门湾1000x5.csv', encoding='unicode_escape')
row = row.fillna(0)
data = np.array(row)

testindex = np.array(pd.read_csv('testindex.csv', encoding='unicode_escape'))
testindex = testindex.reshape(193)
testset = []

for i in testindex:
    testset.append(data[i, :])  # 取出每一行元素 添加到testset中
testset = np.array(testset)

trainindex = np.array(pd.read_csv('trainindex.csv'))
trainindex = trainindex.reshape(770)
trainset = []

for t in trainindex:
    trainset.append(data[i, :])
trainset = np.array(trainset)

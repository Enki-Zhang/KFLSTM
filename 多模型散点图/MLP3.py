#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/22 14:43
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

# 读取 CSV 文件数据
data = pd.read_csv('海门湾label=4(0).csv')

# 将前四列作为协变量 X，最后一列作为因变量 y
X = data.iloc[:, :4].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 构建 Pipeline 对象，用于将 MLP 模型与 GridSearchCV 进行整合
pipe = Pipeline([
    ('model', MLP(hidden_size=10))
])

# 定义超参数空间
param_grid = {
    'model__hidden_size': [5, 10, 20],
    'model__lr': [0.001, 0.01, 0.1]
}

# 初始化 GridSearchCV 对象
grid = GridSearchCV(pipe, param_grid, cv=5)

# 训练模型
grid.fit(X_train, y_train)

# 输出最佳超参数的组合
print('Best hyperparameters:', grid.best_params_)

# 测试模型并绘制散点图
with torch.no_grad():
    y_pred = grid.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()

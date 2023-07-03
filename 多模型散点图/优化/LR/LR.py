#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/30 15:51
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件数据
data = pd.read_csv('海门湾label=4(0).csv')

# 填充空值为0
# data = data.fillna(0)

# 提取协变量和目标变量
X = data.iloc[:, :4].values
y = data.iloc[:, 4].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义线性回归模型
class LR(nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.fc(x)
        return out

# 将PyTorch模型包装在一个scikit-learn兼容的类中
# 将PyTorch模型包装在一个scikit-learn兼容的类中
class TorchEstimator(BaseEstimator):
    def __init__(self, input_size):
        self.input_size = input_size
        self.model = LR(input_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def fit(self, X, y):
        # 转换为Tensor
        X = torch.Tensor(X)
        y = torch.Tensor(y)

        # 训练模型
        for epoch in range(100):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.loss_fn(outputs.squeeze(), y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        # 转换为Tensor
        X = torch.Tensor(X)

        # 预测
        with torch.no_grad():
            outputs = self.model(X)

        return outputs.numpy()

    def score(self, X, y):
        # 预测
        y_pred = self.predict(X)

        # 计算R2得分
        r2 = r2_score(y, y_pred)

        return r2

# 网格搜索和交叉验证优化超参数
model = TorchEstimator(input_size=4)  # 输入特征的维度为4
parameters = {'input_size': [4, 8, 16]}  # 要优化的超参数

grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5)

# 在训练集上进行网格搜索和交叉验证
grid_search.fit(X_train, y_train)

# 打印最优超参数模型
print("Best Hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 计算训练集和测试集上的预测值
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# 创建散点图
plt.scatter(y_train, y_train_pred, label='Train')
plt.scatter(y_test, y_test_pred, label='Test')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

# 打印评价指标
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Train MAE:", train_mae)
print("Test MAE:", test_mae)
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R2 Score:", train_r2)
print("Test R2 Score:", test_r2)

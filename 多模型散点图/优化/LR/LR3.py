#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/30 17:58
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# 读取csv文件
data = pd.read_csv('平滑1000oxy.csv')

# 填充空值为0
data.fillna(0, inplace=True)

# 提取特征和标签
X = data.iloc[:, :4]
y = data.iloc[:, 4]

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 定义超参数范围
param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}

# 创建线性回归模型
model = LinearRegression()

# 执行网格搜索和5折交叉验证
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 获取最佳超参数配置
best_params = grid_search.best_params_
# 使用最佳超参数配置创建最终模型
final_model = LinearRegression(**best_params)

# 在训练集上进行5折交叉验证并获取结果
train_scores = cross_val_score(final_model, X_train, y_train, cv=5)

# 在测试集上进行5折交叉验证并获取结果
test_scores = cross_val_score(final_model, X_test, y_test, cv=5)

# 绘制散点图
plt.scatter(range(1, 6), train_scores, label='Train')
plt.scatter(range(1, 6), test_scores, label='Test')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.legend()
plt.show()
# 绘制散点图
plt.scatter(train_labels, train_predictions, color='blue', label='Training Set')
plt.scatter(labels, predictions, color='red', label='Testing Set')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
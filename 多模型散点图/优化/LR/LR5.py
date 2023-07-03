#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/30 22:04
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件数据
data = pd.read_csv('海门湾label=4(0).csv')

# 填充空值为0
data.fillna(0, inplace=True)

# 提取协变量和目标变量
X = data.iloc[:, :4]
y = data.iloc[:, 4]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义线性回归模型
model = LinearRegression()

# 定义超参数网格
param_grid = {
    'normalize': [True, False],
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'n_jobs': [-1, 1, 2],
    'positive': [True, False],
}

# 网格搜索和交叉验证
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 打印最优超参数
print("Best Parameters: ", grid_search.best_params_)

# 在测试集上进行交叉验证
cv_scores_train = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
cv_scores_test = cross_val_score(grid_search.best_estimator_, X_test, y_test, cv=5)

# 预测训练集和测试集结果
y_pred_train = grid_search.best_estimator_.predict(X_train)
y_pred_test = grid_search.best_estimator_.predict(X_test)

# 计算评估指标
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

# 打印评估指标
print("Train Set Metrics:")
print("MSE:", mse_train)
print("MAE:", mae_train)
print("RMSE:", rmse_train)
print("R2 Score:", r2_train)
print("--------------------")
print("Test Set Metrics:")
print("MSE:", mse_test)
print("MAE:", mae_test)
print("RMSE:", rmse_test)
print("R2 Score:", r2_test)


np.savetxt("LR5_y_pred_train.csv", y_pred_train)
np.savetxt("LR5_y_pred_test.csv", y_pred_test)
np.savetxt("LR5_y_train.csv", y_train)
np.savetxt("LR5_y_test.csv", y_test)
# 绘制散点图
plt.scatter(y_train, y_pred_train, label='Training Set')
plt.scatter(y_test, y_pred_test, label='Testing Set')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

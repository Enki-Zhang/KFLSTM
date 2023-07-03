#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/30 16:30
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 读取CSV文件
data = pd.read_csv('海门湾label=4(0).csv')
data.fillna(0, inplace=True)
# 分割特征和目标变量
X = data.iloc[:, :4]
y = data.iloc[:, 4]

# 空值填充为0
imputer = SimpleImputer(strategy='constant', fill_value=0)
X = imputer.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 定义XGBoost模型
xgb_model = XGBRegressor()

# 设置超参数网格搜索范围
param_grid = {
    'n_estimators': [100, 200, 250],
    'max_depth': [3, 5,7],
    'learning_rate': [0.001,0.01, 0.1, 0.5]
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=10)

# 在训练集上进行网格搜索和交叉验证
grid_search.fit(X_train, y_train)

# 输出最优模型的超参数
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
# 使用最优模型进行预测
y_train_pred = grid_search.predict(X_train)
y_test_pred = grid_search.predict(X_test)
# np.savetxt('y_train_pred.csv', y_train_pred)
# np.savetxt('y_test_pred.csv', y_test_pred)
# np.savetxt('y_train.csv', y_train)
# np.savetxt('y_test.csv', y_test)

# 计算评估指标
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

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




# 绘制散点图
plt.scatter(y_train, y_train_pred, label='Training Set')
plt.scatter(y_test, y_test_pred, label='Test Set')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/30 21:44
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('海门湾label=4(0).csv')

# 空值填充为0
data.fillna(0, inplace=True)

# 提取协变量和目标变量
X = data.iloc[:, :4]
y = data.iloc[:, 4]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义决策树回归模型
cart_model = DecisionTreeRegressor()

# 定义超参数网格
param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20,30,50],
    'min_samples_split': [2, 3, 4, 5, 10, 15, 20,30,40,50],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8,9,10,15,20],
    # 'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    # 'max_features': ['auto', 'sqrt', 'log2'],
    # 'class_weight': [None, 'balanced']
}

# 网格搜索和交叉验证
grid_search = GridSearchCV(cart_model, param_grid, cv=10)
grid_search.fit(X_train, y_train)

# 打印最优超参数
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# 使用最优超参数构建决策树模型
best_cart_model = DecisionTreeRegressor(**best_params)

# 交叉验证预测
y_train_pred = cross_val_predict(best_cart_model, X_train, y_train, cv=10)
y_test_pred = cross_val_predict(best_cart_model, X_test, y_test, cv=10)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 计算指标数据
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("Train MSE:", train_mse)
print("Train MAE:", train_mae)
print("Train RMSE:", train_rmse)
print("Train R2:", train_r2)
print("Test MSE:", test_mse)
print("Test MAE:", test_mae)
print("Test RMSE:", test_rmse)
print("Test R2:", test_r2)

# 绘制散点图
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图

np.savetxt("CART_y_train.csv", y_train)
np.savetxt("CART_y_train_pred.csv", y_train_pred)
np.savetxt("CART_y_test.csv", y_test)
np.savetxt("CART_y_test_pred.csv", y_test_pred)
plt.scatter(y_train, y_train_pred, label='Train')
plt.scatter(y_test, y_test_pred, label='Test')
plt.text(0.75, 0.28, fr'$R^2_{{\mathrm{{train}}}}={train_r2:.2f}$', ha='center', va='center',
         transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style': 'oblique', 'size': 20})
plt.text(0.75, 0.18, fr'$R^2_{{\mathrm{{test}}}}={test_r2:.2f}$', ha='center', va='center',
         transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style': 'oblique', 'size': 20})

lr_label = plt.text(13.42, -0.35,
                    f'(f)CART', ha='right', va='bottom', fontsize=20)
# 设置LR标签背景颜色为灰色
lr_label.set_bbox(dict(facecolor='grey', alpha=0.2, edgecolor='none'))
# 添加图例
# 添加图例
plt.legend()
# 绘制x=y的射线
x = y = range(0, 14)
plt.plot(x, y, color='green', linewidth=1)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.text(14, 14, 'Y=X', color='green', fontsize=20, fontfamily='serif', fontstyle='italic')
plt.xlabel('Observed Values(DO mg/l)', fontsize=20)
plt.ylabel('Predicted Values(DO mg/l)', fontsize=20)
# plt.title('Observed vs. Predicted Values')
plt.tight_layout()  # 去掉图形两边空白
plt.show()

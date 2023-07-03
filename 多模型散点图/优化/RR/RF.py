#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/30 21:11
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 读取CSV文件数据
data = pd.read_csv('海门湾label=4(0).csv')

# 将空值填充为0
data.fillna(0, inplace=True)

# 提取协变量和目标变量
X = data.iloc[:, :4]
y = data.iloc[:, 4]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义随机森林模型
rf_model = RandomForestRegressor()

# 定义超参数网格
param_grid = {
    'n_estimators': [10,20,30,40,50, 60,70,80,90,100, 200],
    'max_depth': [1,2,3,4, 5,6,7,8,9, 10],
    'min_samples_split': [2, 5, 10]
}

# 使用网格搜索和5折交叉验证找到最佳超参数
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 打印最佳超参数模型
best_model = grid_search.best_estimator_
print("Best Model:", best_model)

# 使用5折交叉验证得到训练集的预测结果
train_predictions = cross_val_predict(best_model, X_train, y_train, cv=5)

# 使用5折交叉验证得到测试集的预测结果
test_predictions = cross_val_predict(best_model, X_test, y_test, cv=5)

# 计算训练集和测试集的均方误差
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

# 计算评估指标
mse_train = mean_squared_error(y_train, train_predictions)
mae_train = mean_absolute_error(y_train, train_predictions)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, train_predictions)

mse_test = mean_squared_error(y_test, test_predictions)
mae_test = mean_absolute_error(y_test, test_predictions)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, test_predictions)

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
# 绘制散点图展示真实值和预测值

plt.plot(np.linspace(0, max(y), 100), np.linspace(0, max(y), 100), color='red', linestyle='--')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()

np.savetxt('RF_train_predictions.csv', train_predictions)
np.savetxt('RF_test_predictions.csv', test_predictions)
np.savetxt('RF_y_train.csv', y_train)
np.savetxt('RF_y_test.csv', y_test)
# 绘制散点图
import matplotlib.pyplot as plt
import pandas as pd
# 提取观测值和预测值列
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图
plt.scatter(y_train, train_predictions, label='Train', alpha=0.5)
plt.scatter(y_test, test_predictions, label='Test', alpha=0.5)

# plt.text(0.85, 0.18, fr'$R^2={0.26:.2f}$', ha='center', va='center', transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style':'oblique','size': 20})
# 添加标签
lr_label = plt.text(15.62, -0.362,
                    f'(b)MLP', ha='right', va='bottom',fontsize=20 )
# 设置LR标签背景颜色为灰色
lr_label.set_bbox(dict(facecolor='grey', alpha=0.2, edgecolor='none'))
# 添加图例
plt.legend(loc='upper left')
# 绘制x=y的射线
x = y = range(0, 12)

plt.plot(x, y, color='red', linewidth=1)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.text(9, 10, 'Y=X',color='red',fontsize=20,fontfamily='serif', fontstyle='italic')
plt.xlabel('Observed Values(DO mg/l)',fontsize=20)
plt.ylabel('Predicted Values(DO mg/l)',fontsize=20)
# plt.title('Observed vs. Predicted Values')
plt.tight_layout()  # 去掉图形两边空白
plt.show()



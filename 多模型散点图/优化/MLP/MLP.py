#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/30 15:06
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt



import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import matplotlib.pyplot as plt

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 读取CSV文件数据
data = pd.read_csv('海门湾label=4(0).csv')
# 使用0值填充空值
data.fillna(0, inplace=True)
X = data.iloc[:, :4].values
y = data.iloc[:, 4].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 包装MLP模型为Scikit-Learn估计器
mlp_model = MLP(input_size=4, hidden_size=32, output_size=1)
estimator = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', solver='adam')

# 网格搜索和交叉验证优化超参数
parameters = {'alpha': [0.0001, 0.001, 0.01], 'max_iter': [100, 200, 300,400,500,600,700,],
              # 'learning_rate': ['constant', 'adaptive'],
              'learning_rate_init': [0.001, 0.01, 0.1],
              # 'hidden_layer_sizes': [(32,), (64,), (128,)],
              # 'activation': ['relu', 'tanh', 'logistic'],
              # 'early_stopping': [True],
              }
cv = KFold(n_splits=10, shuffle=True)

grid_search = GridSearchCV(estimator=estimator, param_grid=parameters, cv=cv)
grid_search.fit(X_train, y_train)

# 打印优化的超参数模型
best_model = grid_search.best_estimator_
print("Optimized Model:", best_model)

# 在训练集和测试集上进行预测并绘制散点图
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

plt.scatter(y_train, y_pred_train, label='Train')
plt.scatter(y_test, y_pred_test, label='Test')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

# 计算评价指标
train_mae = mean_absolute_error(y_train, y_pred_train)
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
train_r2 = r2_score(y_train, y_pred_train)

test_mae = mean_absolute_error(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
test_r2 = r2_score(y_test, y_pred_test)
# 打印评价标准
print("训练集上的评价标准:")
print("MAE:", train_mae)
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("R2 Score:", train_r2)

print("\n测试集上的评价标准:")
print("MAE:", test_mae)
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R2 Score:", test_r2)


np.savetxt('MLPLabel2.csv', y_test,)
np.savetxt('MLPPre2.csv', y_pred_test,)
np.savetxt('MLPTrainLabel2.csv', y_train,)
np.savetxt('MLPTrainPre2.csv', y_pred_train,)
# 绘制散点图
import matplotlib.pyplot as plt
import pandas as pd
# 提取观测值和预测值列
# observed = pd.read_csv('MLPLabel.csv')
# predicted = pd.read_csv('MLPPre.csv')
# predicted = pd.read_csv('MLPTrainPre.csv')
# predictedKFLSTM = pd.read_csv('KFLSTMPRE.csv')
# predictedKFGRU = pd.read_csv('KFGRUPRE1.csv')
# predictedGRU = pd.read_csv('GRUPRE1.csv')
# predictedXGB = pd.read_csv('xgb.csv')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图
# plt.scatter(observed[:1000], train[:1000], label='train',marker='o', color='red')
plt.scatter(y_train, y_pred_train, label='Train set',marker='o', color='red')
plt.scatter(y_test, y_pred_test, label='Test set',marker='^', color='blue')
# plt.scatter(observed, predictedLSTM, label='LSTM-seq2seq',marker='o',color='blue')
# plt.scatter(observed, predictedKFLSTM, label='KF-LSTM',marker='s',color='green')
# plt.scatter(observed, predictedKFGRU, label='KF-GRU',marker='d',color='purple')
# plt.scatter(observed, predictedXGB, label='XGboost',marker='d',color='#018101')
plt.text(0.85, 0.18, fr'$R^2={0.26:.2f}$', ha='center', va='center', transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style':'oblique','size': 20})
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




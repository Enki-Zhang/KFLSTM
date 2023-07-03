#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/22 15:23
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 读入数据
data = pd.read_csv('海门湾label=4(0).csv')
X = data.iloc[:, :4].values
y = data.iloc[:, 4].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义模型
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 网格搜索参数优化
input_dim = 4
output_dim = 1
hidden_dim_list = [5, 10, 15]
lr_list = [0.001, 0.01, 0.1]

best_model = None
best_loss = float('inf')
for hidden_dim in hidden_dim_list:
    for lr in lr_list:
        model = FFNN(input_dim, hidden_dim, output_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        num_epochs = 1000
        for epoch in range(num_epochs):
            inputs = torch.Tensor(X_train.astype(float))
            labels = torch.Tensor(y_train.astype(float)).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 测试模型
        inputs_test = torch.Tensor(X_test.astype(float))
        labels_test = torch.Tensor(y_test.astype(float)).unsqueeze(1)
        outputs_test = model(inputs_test)
        loss_test = criterion(outputs_test, labels_test).item()

        # 保存最佳模型
        if loss_test < best_loss:
            best_loss = loss_test
            best_model = model

print(f'Best loss: {best_loss:.4f}')

# 预测并画图
inputs_all = torch.Tensor(X.astype(float))
labels_all = torch.Tensor(y.astype(float))
outputs_all = best_model(inputs_all)

plt.scatter(labels_all, outputs_all.detach().numpy())
plt.plot([0, np.max(labels_all.numpy())], [0, np.max(outputs_all.detach().numpy())], 'r--')

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# 输出 R2 分数
print(f'R2 score: {r2_score(y, outputs_all.detach().numpy()):.4f}')
# 将预测值和真实值保存到 csv 文件中
results_df = pd.DataFrame({
                           'Predictions': outputs_all.detach().numpy().flatten()})
results_df.to_csv('FFNNPre.csv', index=False)

true_df = pd.DataFrame({'True Values': y})
true_df.to_csv('FFNNTrue.csv', index=False)
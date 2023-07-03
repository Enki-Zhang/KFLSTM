#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/22 14:36
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取csv文件数据
data = pd.read_csv('海门湾label=4(0).csv')

# 将数据拆分为80%训练集和20%测试集
train_data, test_data = train_test_split(data, test_size=0.2)

# 获取协变量和响应变量
train_x = train_data.iloc[:, :4].values
train_y = train_data.iloc[:, 4].values
test_x = test_data.iloc[:, :4].values
test_y = test_data.iloc[:, 4].values


# 定义模型类
class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 实例化模型和损失函数
model = FFNN()
criterion = nn.MSELoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 将数据转换为张量
    inputs = torch.Tensor(train_x)
    targets = torch.Tensor(train_y)

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs.squeeze(), targets)

    # 反向传播和优化器步骤
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 在测试集上进行预测
test_inputs = torch.Tensor(test_x)
predictions = model(test_inputs).squeeze().detach().numpy()

# 显示拟合效果
import matplotlib.pyplot as plt

plt.scatter(test_y, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# 添加图例
plt.legend()
plt.scatter(test_y, predictions, label='FFNN',marker='^', color='blue')
# 绘制x=y的射线
x = y = range(0, 12)
plt.plot(x, y, color='red', linewidth=1)
plt.text(9, 10, 'Y=X',color='red',fontsize=20,fontfamily='serif', fontstyle='italic')
plt.xlabel('Observed Values(DO)')
plt.ylabel('Predicted Values(DO)')
# plt.title('Observed vs. Predicted Values')
plt.tight_layout()  # 去掉图形两边空白
plt.show()

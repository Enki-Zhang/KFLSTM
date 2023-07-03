import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 读取csv文件
data = pd.read_csv('海门湾label=4(0).csv')

# 将数据分为协变量和目标变量
X = data.iloc[:, :4].values
y = data.iloc[:, 4].values

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型、优化器和损失函数
model = MLP(input_size=4, hidden_size=8, output_size=1)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 定义超参数
epochs = 100

# 将数据转换为tensor的形式
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).float()

# 记录训练过程中的损失函数值
train_loss = []

# 训练模型
for epoch in range(epochs):
    # 前向传播
    y_pred = model(X_train_tensor)

    # 计算损失函数
    loss = criterion(y_pred, y_train_tensor.unsqueeze(1))

    # 反向传播和参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失函数值
    train_loss.append(loss.item())

    # 打印训练过程中的损失函数值
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 将测试数据转换为tensor的形式
X_test_tensor = torch.tensor(X_test).float()

# 进行预测
y_pred = model(X_test_tensor).detach().numpy()

# 绘制拟合效果图
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Fitting Result')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
# 读取CSV文件
data = pd.read_csv('平滑1000oxy.csv')

# 填充空值为0
data.fillna(0, inplace=True)

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]
def fit_arima_model(train, test):
    # 提取协变量列作为特征
    features = train.iloc[:, :4]

    # 提取目标列作为目标变量
    target = train.iloc[:, 4]

    # 拟合ARIMA模型
    model = ARIMA(target, order=(2, 1, 2))  # 这里的参数是示例，请根据实际情况进行调整
    model_fit = model.fit()

    # 预测训练集和测试集的结果
    train_predictions = model_fit.predict(start=1, end=len(train))
    test_predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

    return train_predictions, test_predictions
# 拟合ARIMA模型并获取预测结果
train_predictions, test_predictions = fit_arima_model(train_data, test_data)
np.savetxt('train_predictions.csv', train_predictions)
np.savetex('test_predictions.csv', test_predictions)
np.savetext('train_data.csv', train_data)
np.savetext('test_data.csv', test_data)
# 绘制散点图
plt.scatter(range(1, len(train_data) + 1), train_data.iloc[:, 4], color='blue', label='Train')
plt.scatter(range(len(train_data), len(train_data) + len(test_data)), test_data.iloc[:, 4], color='red', label='Test')
plt.plot(range(1, len(train_data) + 1), train_predictions, color='green', label='Train Predictions')
plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_predictions, color='orange', label='Test Predictions')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ARIMA Model Predictions')
plt.show()

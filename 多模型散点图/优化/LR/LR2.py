import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('海门湾label=4(0).csv')

# 填充空值为0
data.fillna(0, inplace=True)

# 提取自变量和因变量
X = data.iloc[:, :4]
y = data.iloc[:, 4]

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义超参数网格
param_grid = {'fit_intercept': [True, False]}

# 创建线性回归模型
lr = LinearRegression()

# 使用网格搜索和5折交叉验证找到最佳超参数
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 打印最佳超参数和模型评分
print("Best Parameters: ", grid_search.best_params_)
print("Train R^2 Score: ", grid_search.best_score_)

# 在测试集上评估模型
test_r2_score = grid_search.score(X_test, y_test)
print("Test R^2 Score: ", test_r2_score)

# 绘制散点图展示训练集和测试集上的预测结果
plt.scatter(range(len(y_train)), y_train, label='Training Set')
plt.scatter(range(len(y_test)), y_test, label='Test Set')
plt.scatter(range(len(y_test)), grid_search.predict(X_test), label='Predictions')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Linear Regression Predictions')
plt.show()

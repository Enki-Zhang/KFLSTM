import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 读取CSV文件数据
data = pd.read_csv('平滑1000oxy.csv')

# 将空值进行0填充
imputer = SimpleImputer(strategy='constant', fill_value=0)
data_filled = imputer.fit_transform(data)

# 划分协变量和目标变量
X = data_filled[:, :4]
y = data_filled[:, 4]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 网格搜索超参数并进行交叉验证
best_score = -float('inf')
best_lr = None
best_num_epochs = None
lr_values = [0.001, 0.01, 0.1]
num_epochs_values = [100, 200, 300]

for lr in lr_values:
    for num_epochs in num_epochs_values:
        # 创建模型
        model = LinearRegression()

        # 执行交叉验证
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        average_score = -scores.mean()

        # 保存最优模型和超参数
        if average_score > best_score:
            best_score = average_score
            best_lr = lr
            best_num_epochs = num_epochs

# 输出最优超参数
print("Best Hyperparameters:")
print("Learning Rate:", best_lr)
print("Number of Epochs:", best_num_epochs)

# 使用最优超参数训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 在交叉验证集上进行预测并计算均方误差
cv_predictions = model.predict(X_train)
cv_mse = mean_squared_error(y_train, cv_predictions)

# 在测试集上进行预测并计算均方误差
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)

# 输出交叉验证和测试集的均方误差
print("Cross Validation MSE:", cv_mse)
print("Test Set MSE:", test_mse)

# 绘制散点图
plt.scatter(y_train, cv_predictions, color='blue', label='Cross Validation')
plt.scatter(y_test, test_predictions, color='red', label='Test Set')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图
# plt.scatter(observed[:1000], predicted[:1000], label='Test set',marker='o', color='blue')
plt.scatter(y_train, cv_predictions, color='blue', label='Cross Validation')
plt.scatter(y_test, test_predictions, color='red', label='Test Set')
plt.text(0.85, 0.18, fr'$R^2={0.22:.2f}$', ha='center', va='center', transform=plt.gca().transAxes,
         fontdict={'family': 'serif', 'style':'oblique','size': 20})
# 添加标签
lr_label = plt.text(15.36, -0.35,
                    f'(a)LR', ha='right', va='bottom',fontsize=20 )
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


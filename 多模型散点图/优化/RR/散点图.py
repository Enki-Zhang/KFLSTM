#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/7/2 9:49
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/7/2 9:45
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/7/1 23:31

# 绘制散点图
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# 绘制散点图
y_train = pd.read_csv('RF_y_train.csv')
y_train_pred = pd.read_csv('RF_train_predictions.csv')
y_test = pd.read_csv('RF_y_test.csv')
y_test_pred = pd.read_csv('RF_test_predictions.csv')
plt.scatter(y_train[:1000], y_train_pred[:1000], label='Train set',color='red')
plt.scatter(y_test[:1000], y_test_pred[:1000], label='Test set',color='blue')
plt.text(0.82, 0.28, fr'$R^2_{{\mathrm{{train}}}}={0.68:.2f}$',ha='center', va='center',
         transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style':'oblique','size': 20})
plt.text(0.82, 0.18, fr'$R^2_{{\mathrm{{test}}}}={0.67:.2f}$',ha='center', va='center',
         transform=plt.gca().transAxes, fontdict={'family': 'serif', 'style':'oblique','size': 20})

lr_label = plt.text(15.60, -0.49,
                    f'(d)RF', ha='right', va='bottom',fontsize=20 )
# 设置LR标签背景颜色为灰色
lr_label.set_bbox(dict(facecolor='grey', alpha=0.2, edgecolor='none'))
# 添加图例
# 添加图例
plt.legend()
# 绘制x=y的射线
x = y = range(0, 16)
plt.plot(x, y, color='green', linewidth=1)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.text(13, 14, 'Y=X',color='green',fontsize=20,fontfamily='serif', fontstyle='italic')
plt.xlabel('Observed Values(DO mg/l)',fontsize=20)
plt.ylabel('Predicted Values(DO mg/l)',fontsize=20)
# plt.title('Observed vs. Predicted Values')
plt.tight_layout()  # 去掉图形两边空白
plt.show()

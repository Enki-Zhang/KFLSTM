# -- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:42:07 2021

@author: Administrator
"""
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6), dpi=200)
ax = Axes3D(fig)

# 数据录入
Y = np.array([2, 4, 8, 16, 32, 64, 128])
X = np.array([0.001, 0.01, 0.1])
X, Y = np.meshgrid(X, Y)
print("网格化后的X=", X)
print("X维度信息", X.shape)
print("网格化后的Y=", Y)
print("Y维度信息", Y.shape)

Z = np.array(
    [
        [0.85, 0.77, 0.75, 0.86, 0.96, 1.14, 1.22],
        [1.52, 1.21, 1.29, 1.16, 1.26, 1.19, 1.15],
        [1.56, 1.73, 1.72, 1.59, 1.62, 1.61, 1.60],
    ]
)
print("维度调整前的Z轴数据维度", Z.shape)
Z = Z.T
print("维度调整后的Z轴数据维度", Z.shape)

# 绘制三维曲面图
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer')
# 设置三个坐标轴信息
ax.set_xlabel('learning rate', color='black')
ax.set_ylabel('batch_size', color='black')
ax.set_zlabel('RMSE', color='black')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.tight_layout(pad=0.4, w_pad=100, h_pad=1.0)
plt.draw()
plt.show()
plt.savefig('3D.jpg', bbox_inches='tight')

# fig = plt.figure()
# ax = Axes3D(fig)
# x = [2,2,2,4,4,4,8,8,8,16,16,16,32,32,32,64,64,64,128,128,128]
# y = [0.001, 0.01, 0.1,0.001, 0.01, 0.1,0.001, 0.01, 0.1,0.001, 0.01, 0.1,0.001, 0.01, 0.1,0.001, 0.01, 0.1,0.001, 0.01, 0.1]
# z = np.array([[0.85, 1.52, 1.56, 0.77, 1.21, 1.73, 0.75, 1.29, 1.72, 0.86, 1.16, 1.59, 0.96, 1.26, 1.62, 1.14, 1.19, 1.61, 1.22, 1.15, 1.6]])
# x, y = np.meshgrid(x, y)
# ax.plot_surface(x,y,z, rstride=0.001, cstride=0.001, cmap='rainbow')
# plt.show()


# fig = plt.figure()
# ax = Axes3D(fig)

# X = [2,2,2,4,4,4,8,8,8,16,16,16,32,32,32]
# Y = [0.001,0.01,0.1,0.001,0.01,0.1,0.001,0.01,0.1,0.001,0.01,0.1,0.001,0.01,0.1]
# X, Y = np.meshgrid(X, Y)
# Z = np.array([[0.85,1.52,1.56,0.77,1.21,1.73, 0.75,1.29,1.72,0.86,1.16,1.59,0.96,1.26,1.62]])

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

# plt.draw()
# plt.pause(10)
# plt.savefig('3D.jpg')
# plt.close()

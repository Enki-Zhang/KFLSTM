#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Enki
# Time : 2023/6/22 15:06
import pandas as pd
from sklearn.utils import resample

# 读取CSV文件
df = pd.read_csv('MLPPre.csv')

# 进行过采样
df_resampled = resample(df, replace=True, n_samples=2626)

# 输出过采样后的数据集大小
print("过采样后的数据集大小：", df_resampled.shape)
# 保存过采样后的数据集为CSV文件
df_resampled.to_csv('MLPTrainPre.csv', index=False)
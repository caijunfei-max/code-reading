# -*-coding:utf-8 -*-
"""
# File       : dataTesting.py
# Time       ：2023/6/30 9:47
# version    : 
# Author: Jun_fei Cai
# Description: 
"""

import numpy as np
import pandas as pd
from functionCjf import *
from torch.utils.data import DataLoader


all_data = pd.read_csv('../Database.csv', header=0).iloc[:, 1:19].to_numpy()
raw_x = all_data[:696, :6]
raw_y = all_data[:696, 17].reshape(-1, 1)
array_x_y = np.column_stack((raw_x, raw_y))
# reshape的-4参数代表-1占据的这个参数将由其他参数自动计算
# 比如这里应该是(693, 1)，-1意思是根据原来的数据和后面的1列来计算出新的numpy.array的形状。
dataset = FeatureDataset(raw_x, raw_y)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
# dataloader打包dataset中的数据为可迭代对象，并且分成若干份，每份中有20个数据（batch的大小）

print(dataloader[0])
print(dataloader)
# for i, data in enumerate(dataloader):
#     # print(type(i),type(data))
#     print(i, data)

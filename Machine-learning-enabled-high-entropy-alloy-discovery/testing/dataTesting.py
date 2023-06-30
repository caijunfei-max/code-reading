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


all_data = pd.read_csv('../Database.csv', header=0).iloc[:, 1:19].to_numpy()
raw_x = all_data[:696, :6]
raw_y = all_data[:696, 17].reshape(-1, 1)
# reshape的-1参数代表-1占据的这个参数将由其他参数自动计算
# 比如这里应该是(696, 1)，-1意思是根据原来的数据和后面的1列来计算出新的numpy.array的形状。

print(raw_x.shape)


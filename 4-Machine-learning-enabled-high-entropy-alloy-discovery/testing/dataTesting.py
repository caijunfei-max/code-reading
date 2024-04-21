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
import torch
import torch.nn as nn
from functionCjf import *
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


all_data = pd.read_csv('../Database.csv', header=0).iloc[:, 1:19].to_numpy()
raw_x = all_data[:696, :6]
raw_y = all_data[:696, 17].reshape(-1, 1)
array_x_y = np.column_stack((raw_x, raw_y))
# reshape的-4参数代表-1占据的这个参数将由其他参数自动计算
# 比如这里应该是(693, 1)，-1意思是根据原来的数据和后面的1列来计算出新的numpy.array的形状。
dataset = FeatureDataset(raw_x, raw_y)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
# dataloader打包dataset中的数据为可迭代对象，并且分成若干份，每份中有20个数据（batch的大小）,内部每一个batch都会用一个数值索引


class WAE(nn.Module):
    def __init__(self, input_size):
        super(WAE, self).__init__()
        self.input_size = input_size

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Linear(80, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, 2),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Linear(80, self.input_size),
            nn.Softmax(dim=1)  # (softmax along dimension 1)
        )
        self.apply(weights_init)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

model = WAE(raw_x.shape[1]).to(device)


# for i, data in enumerate(dataloader):
#     # print(type(i),type(data))
#     print(i, data)

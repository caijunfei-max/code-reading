# -*-coding:utf-8 -*-
"""
# File       : test.py
# Time       ：2023/7/7 10:40
# version    : 
# Author: Jun_fei Cai
# Description: A testing script
"""

import torch.nn as nn
from Functions import *
import pandas as pd

root = '/content/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                        )   # 3层，神经元数量分别是80，64，48，每一层神经层后面都有标准化层以及一个激活函数层

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
                        nn.Softmax(dim=1)   # (softmax along dimension 1)
                        )  # 解码模型和编码模型的结构相同，但是层的作用相反
        self.apply(weights_init)  # applying the initialization of weight and bias

    def forward(self, x):
        z = self._encode(x)        # 编码数据
        x_recon = self._decode(z)  # x_recon重新解码后的数据

        return x_recon, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


params = {
    'num_epoch': 200,
    'batch_size': 20,
    'lr': 5e-4,
    'weight_decay': 0.0,
    'sigma': 8.0,
    'MMD_lambda': 1e-4,
    'model_name': 'WAE_v1'}
all_data = pd.read_csv('../data_base.csv', header=0).iloc[:, 1:19].to_numpy()
raw_x = all_data[:696, :6]
raw_y = all_data[:696, 17].reshape(-1, 1)
dataset = FeatureDataset(raw_x[:], raw_y[:])
dataloader = DataLoader(dataset, batch_size=params['batch_size'])
model = WAE(raw_x.shape[1]).to(device)

for i, data in enumerate(dataloader):
    x = data[0].to(device)
    model.train()
    recon_x, z_tilde = model(x)
    print(recon_x, z_tilde)


# -*-coding:utf-8 -*-
"""
# File       : functionCjf.py
# Time       ：2023/6/30 10:08
# version    : 
# Author: Jun_fei Cai
# Description: 
"""
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # 初始化类,且必须输入x, y两个参数

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])
    # __getitem__用于帮助实例化类能够使用索引进行提取参数


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

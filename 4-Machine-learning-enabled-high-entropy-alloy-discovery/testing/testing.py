# -*-coding:utf-8 -*-
"""
# File       : testing.py
# Time       ：2023/7/6 9:34
# version    : 
# Author: Jun_fei Cai
# Description: 
"""
import torch
from torch.utils.data import Dataset


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # 权重初始化为正态分布，1.0和0.02为初始化分布的平均值和标准差
        m.bias.data.fill_(0)  # bias初始化为0


# -*-coding:utf-8 -*-
'''
# File       : filevisual.py
# Time       ：2023/6/15 9:15
# version    : 
# Author: Junfei Cai
# Description: This script is written for checking the data from .pkl file
'''

import pickle
path_1 = "data/bandgap_training_data.pkl"
path_2 ="data/element_data.pkl"
path_3 ="data/training_compounds.pkl"

f1 = open(path_1,'rb')
data_1 = pickle.load(f1)
f2 = open(path_2,'rb')
data_2 = pickle.load(f2)
f3 = open(path_3,'rb')
data_3 = pickle.load(f3)

print(data_2)

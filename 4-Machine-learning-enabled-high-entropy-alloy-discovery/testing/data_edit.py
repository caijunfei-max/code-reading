# -*-coding:utf-8 -*-
"""
# File       : data_edit.py
# Time       ：2023/6/30 8:23
# version    : 
# Author: Junfei Cai
# Description: 
"""

import pandas as pd


def xlsx_to_pd():
    data_xls = pd.read_excel('../Data_base.xlsx', index_col=0)
    data_xls.to_csv('../data_base.csv', encoding="utf-8")



xlsx_to_pd()

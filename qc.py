# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:42:21 2023

@author: 86187
"""
import numpy as np
import seaborn as sns
import pandas as pd
A = pd.read_csv(r'D:\qc\轴承故障数据集\Mechanical-datasets-master\gearbox\gearset\Surface_30_2.csv',delimiter=('\t'), header=None)
A.to_csv(r'D:\qc\轴承故障数据集\Mechanical-datasets-master\gearbox\gearset\datasetgearset\Surface_30_2.csv', header=None,index=False)
import scipy.io as scio

# 读取mat文件
from matplotlib import pyplot as plt

# data = scio.loadmat('./datanew/0.mat')
#
# # 获取data和cam
# data_array = data['data']
# cam_array = data['cam']
#
# # 绘制热力图
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# sns.heatmap(cam_array, cmap='coolwarm')
# plt.show()

# 插值

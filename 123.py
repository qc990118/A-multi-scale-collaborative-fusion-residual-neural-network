import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat_file_path = 'data/Dataset_A/1.mat'
# 加载MAT文件
mat_data = scipy.io.loadmat(mat_file_path)
data = loadmat(mat_file_path)  # 读取mat文件
print(data.keys())
# 获取振动数据，假设MAT文件中的振动数据存储在变量名为 "vibration_data" 中
vibration_data = mat_data['X118_DE_time']
# 仅使用前2000个数据点
num_data_points_to_plot = 5000

# 创建波形图，只展示前2000个数据点，不显示坐标轴
plt.figure()
plt.plot(vibration_data[:num_data_points_to_plot])
plt.axis('off')  # 关闭坐标轴显示
plt.show()

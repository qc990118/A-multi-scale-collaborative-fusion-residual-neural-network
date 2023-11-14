import scipy
from scipy.io import loadmat
import pandas as pd
import scipy.io as sio
# df = pd.read_csv('./data1/20_0/0.csv')
# print(df.keys())
# # 选择第二列数据
# a= df.iloc[15:, 1]
# # 打印第二列数据
# print(a)
# sio.savemat('./data3/20_0/0.mat', {'value': a})
from scipy.io import loadmat
import scipy.io as sio
path = r"datanew/9.mat"  # mat文件路径
data = loadmat(path)  # 读取mat文件
print(data.keys())  # 查看mat文件中包含的变量
# 读取某个键值下的数据
mat_file = scipy.io.loadmat('datanew/9.mat')
data = mat_file['cam']
data1 = mat_file['data']
import os
import xlsxwriter
# 打印数据
print(data1)
# # 将数据保存到Excel文件中
# df = pd.DataFrame(data)
# # create directory if it doesn't exist
# if not os.path.exists('D:/qc/第二篇小论/train'):
#     os.makedirs('D:/qc/第二篇小论/train')
#
# # create workbook and worksheet
# workbook = xlsxwriter.Workbook('D:/qc/第二篇小论/train/data.xlsx')
# worksheet = workbook.add_worksheet()
#
# # write data to worksheet
# for i in range(len(data)):
#     for j in range(len(data[i])):
#         worksheet.write(i, j, data[i][j])
#
# # close workbook
# workbook.close()
# # 将DataFrame写入Excel文件
# writer = pd.ExcelWriter('D:/qc/第二篇小论/train/data.xlsx', engine='xlsxwriter')
# df.to_excel(writer, sheet_name='Sheet1', index=False)
# writer.save()


0



import numpy as np
import torch
import os
import re
import scipy.io as scio
import scipy.signal
from torch.utils import data as da
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

raw_num = 100
class Data(object):
    def __init__(self):
        self.data = self.get_data()
        self.label = self.get_label()
    def file_list(self):
        return os.listdir('./data/Dataset_B')
    def get_data(self):
        file_list = self.file_list()
        x = np.zeros((1024, 0))
        print(len(file_list))
        print(x)
        for i in range(len(file_list)):
            file = scio.loadmat('./data/Dataset_B/{}'.format(file_list[i]))
            for k in file.keys():
                file_matched = re.match('X\d{3}_DE_time', k)
                if file_matched:
                    key = file_matched.group()
            data1 = np.array(file[key][0: 102400]) #0:80624
            print(len(data1))
            for j in range(0, len(data1)-1023, 1024):
                  x = np.concatenate((x, data1[j:j+1024]), axis=1)
        return x.T
    def get_label(self):
        file_list = self.file_list()
        title = np.array([i.replace('.mat', '') for i in file_list])
        label = title[:, np.newaxis]
        label_copy = np.copy(label)
        for _ in range(raw_num-1):
            label = np.hstack((label, label_copy))
        return label.flatten()
def add_gaussian_noise(data, snr):
    if snr is None:
        return data
    else:
        power = np.sum(data ** 2) / len(data)
        noise_power = power / (10 ** (snr / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(data))
        noisy_data = data + noise
        return noisy_data
Data = Data()
data = Data.data
label = Data.label
y = label.astype("int32")
Data_with_noise = []
for i in range(len(data)):
    noisy_data = add_gaussian_noise(data[i], snr=-2)
    Data_with_noise.append(noisy_data)
Data_with_noise = np.asarray(Data_with_noise)
ss = MinMaxScaler()
Data_with_noise = Data_with_noise.T
Data_with_noise = ss.fit_transform(Data_with_noise).T
# ss = MinMaxScaler()
# data = data.T
# data = ss.fit_transform(data).T
X_train, X_test, y_train, y_test = train_test_split(Data_with_noise, y, test_size=0.3, random_state=2, stratify=y)
X_train = torch.from_numpy(X_train).unsqueeze(1)
X_test = torch.from_numpy(X_test).unsqueeze(1)
class TrainDataset(da.Dataset):
    def __init__(self):
        self.Data = X_train
        self.Label = y_train
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
class TestDataset(da.Dataset):
    def __init__(self):
        self.Data = X_test
        self.Label = y_test
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
Train = TrainDataset()
print(len(Train))
Test = TestDataset()
# for data in Test:
#     txt, label = data
#     print(txt.shape, label)
train_loader = da.DataLoader(Train, batch_size=32, shuffle=True)
test_loader = da.DataLoader(Test, batch_size=10, shuffle=False)
print(f"test_loader batch_size: {test_loader.batch_size}")
print(f"test_loader length: {len(test_loader)}")
print(len(test_loader.dataset))


# class Data(object):
#
#     def __init__(self, data_dir):
#         self.data = self.get_data(data_dir)
#         self.label = self.get_label(data_dir)
#
#     def file_list(self, data_dir):
#         return os.listdir(data_dir)
#
#     def get_data(self, data_dir):
#         file_list = self.file_list(data_dir)
#         x = np.zeros((1024, 0))
#         print(len(file_list))
#         print(x)
#         for i in range(len(file_list)):
#             file = scio.loadmat(os.path.join(data_dir, file_list[i]))
#             for k in file.keys():
#                 file_matched = re.match('X\d{3}_DE_time', k)
#                 if file_matched:
#                     key = file_matched.group()
#             data1 = np.array(file[key][0: 102400]) #0:80624
#             print(len(data1))
#             for j in range(0, len(data1)-1023, 1024):
#                   x = np.concatenate((x, data1[j:j+1024]), axis=1)
#         return x.T
#
#     def get_label(self, data_dir):
#         file_list = self.file_list(data_dir)
#         title = np.array([i.replace('.mat', '') for i in file_list])
#         label = title[:, np.newaxis]
#         raw_num = 100
#         label_copy = np.copy(label)
#         for _ in range(raw_num-1):
#             label = np.hstack((label, label_copy))
#         return label.flatten()
#
# class Data1(object):
#     raw_num = 30
#     def __init__(self, data_dir):
#         self.data = self.get_data(data_dir)
#         self.label = self.get_label(data_dir)
#
#     def file_list(self, data_dir):
#         return os.listdir(data_dir)
#
#     def get_data(self, data_dir):
#         file_list = self.file_list(data_dir)
#         x = np.zeros((1024, 0))
#         print(len(file_list))
#         print(x)
#         for i in range(len(file_list)):
#             file = scio.loadmat(os.path.join(data_dir, file_list[i]))
#             for k in file.keys():
#                 file_matched = re.match('X\d{3}_DE_time', k)
#                 if file_matched:
#                     key = file_matched.group()
#             data1 = np.array(file[key][0: 30720]) #0:80624
#             print(len(data1))
#             for j in range(0, len(data1)-1023, 1024):
#                   x = np.concatenate((x, data1[j:j+1024]), axis=1)
#         return x.T
#
#     def get_label(self, data_dir):
#         file_list = self.file_list(data_dir)
#         title = np.array([i.replace('.mat', '') for i in file_list])
#         label = title[:, np.newaxis]
#         label_copy = np.copy(label)
#         raw_num=30
#         for _ in range(raw_num-1):
#             label = np.hstack((label, label_copy))
#         return label.flatten()
#
# data_dir_train = './data/Dataset_B'
# data_dir_test = './data/Dataset_A'
#
# Data_train = Data(data_dir_train)
# Data_test = Data1(data_dir_test)
#
# data_train = Data_train.data
# label_train = Data_train.label.astype("int32")
# data_test = Data_test.data
# label_test = Data_test.label.astype("int32")
#
# Data_with_noise_train = []
# Data_with_noise_test = []
# def add_gaussian_noise(data, snr):
#     if snr is None:
#         return data
#     else:
#         power = np.sum(data ** 2) / len(data)
#         noise_power = power / (10 ** (snr / 10.0))
#         noise = np.random.normal(0, np.sqrt(noise_power), len(data))
#         noisy_data = data + noise
#         return noisy_data
# for i in range(len(data_train)):
#     noisy_data = add_gaussian_noise(data_train[i],snr=None)
#     Data_with_noise_train.append(noisy_data)
#
# for i in range(len(data_test)):
#     noisy_data = add_gaussian_noise(data_test[i],snr=None)
#     Data_with_noise_test.append(noisy_data)
#
# Data_with_noise_train = np.asarray(Data_with_noise_train)
# Data_with_noise_test = np.asarray(Data_with_noise_test)
#
# ss = MinMaxScaler()
#
# Data_with_noise_train = Data_with_noise_train.T
# Data_with_noise_test = Data_with_noise_test.T
#
# Data_with_noise_train = ss.fit_transform(Data_with_noise_train).T
# Data_with_noise_test = ss.fit_transform(Data_with_noise_test).T
#
# X_train = torch.from_numpy(Data_with_noise_train).unsqueeze(1)
# X_test = torch.from_numpy(Data_with_noise_test).unsqueeze(1)
#
# class TrainDataset(da.Dataset):
#     def __init__(self):
#         self.Data = X_train
#         self.Label = label_train
#
#     def __getitem__(self, index):
#         txt = self.Data[index]
#         label = self.Label[index]
#         return txt, label
#
#     def __len__(self):
#         return len(self.Data)
#
# class TestDataset(da.Dataset):
#     def __init__(self):
#         self.Data = X_test
#         self.Label = label_test
#
#     def __getitem__(self, index):
#         txt = self.Data[index]
#         label = self.Label[index]
#         return txt, label
#
#     def __len__(self):
#         return len(self.Data)
#
# Train = TrainDataset()
# print(len(Train))
# Test = TestDataset()
# train_loader = da.DataLoader(Train, batch_size=48, shuffle=True)
# test_loader = da.DataLoader(Test, batch_size=10, shuffle=True)
# print(f"test_loader batch_size: {test_loader.batch_size}")
# print(f"test_loader length: {len(test_loader)}")
# print(len(train_loader.dataset))
# print(len(test_loader.dataset))


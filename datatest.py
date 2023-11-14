# from scipy.io import loadmat
# import pandas as pd
#
# df = pd.read_csv('./data1/20_0/0.csv')
# print(df.keys())
# # 选择第二列数据
# col_2 = df.iloc[15:, 1]
# # 打印第二列数据
# print(col_2)
# from scipy.io import loadmat
# import scipy.io as sio
# path = r"data2/600/0.mat"  # mat文件路径
# data = loadmat(path)  # 读取mat文件
# print(data.keys())  # 查看mat文件中包含的变量
# import scipy.io as sio
#
# # 读取.mat文件
# mat_file = sio.loadmat('data2/600/0.mat')
#
# # 读取指定键值下的值
# data = mat_file['data']
#
# # 列出数据的行数
# print(f"Rows: {data.shape[0]}")
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

# raw_num = 100
# class Data(object):
#     def __init__(self):
#         self.data = self.get_data()
#         self.label = self.get_label()
#     def file_list(self):
#         return os.listdir('./dataseu/30_2')
#     def get_data(self):
#         file_list = self.file_list()
#         x = np.zeros((2048, 0))
#         print(len(file_list))
#         print(x)
#         for i in range(len(file_list)):
#             file = scio.loadmat('./dataseu/30_2/{}'.format(file_list[i]))
#             for k in file.keys():
#                 file_matched = re.match('data', k)
#                 if file_matched:
#                     key = file_matched.group()
#             data1 = np.array(file[key][0: 204800]) #0:80624
#             print(len(data1))
#             for j in range(0, len(data1)-2047, 2048):
#                   x = np.concatenate((x, data1[j:j+2048]), axis=1)
#         return x.T
#     def get_label(self):
#         file_list = self.file_list()
#         title = np.array([i.replace('.mat', '') for i in file_list])
#         label = title[:, np.newaxis]
#         label_copy = np.copy(label)
#         for _ in range(raw_num-1):
#             label = np.hstack((label, label_copy))
#         return label.flatten()
import os
import numpy as np
import pandas as pd

raw_num = 100

class Data(object):
    def __init__(self):
        self.data = self.get_data()
        self.label = self.get_label()

    def file_list(self):
        return os.listdir('./datatest/30_2')

    def get_data(self):
        file_list = self.file_list()
        x = np.zeros((2048, 0))
        print(len(file_list))
        print(x)
        for i in range(len(file_list)):
            file_path = './datatest/30_2/{}'.format(file_list[i])
            if file_path.endswith('.csv'):  # Read only CSV files
                data1 = pd.read_csv(file_path).values.flatten()[:204800]
                print(len(data1))
                for j in range(0, len(data1)-2047, 2048):
                    x = np.concatenate((x, data1[j:j+2048].reshape(-1, 1)), axis=1)
        return x.T

    def get_label(self):
        file_list = self.file_list()
        title = np.array([i.replace('.csv', '') for i in file_list if i.endswith('.csv')])
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
    noisy_data = add_gaussian_noise(data[i], snr=-4)
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
print(len(train_loader.dataset))
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
#         x = np.zeros((2048, 0))
#         print(len(file_list))
#         print(x)
#         for i in range(len(file_list)):
#             file = scio.loadmat(os.path.join(data_dir, file_list[i]))
#             for k in file.keys():
#                 file_matched = re.match('data', k)
#                 if file_matched:
#                     key = file_matched.group()
#             data1 = np.array(file[key][0: 204800]) #0:80624
#             print(len(data1))
#             for j in range(0, len(data1)-2047, 2048):
#                   x = np.concatenate((x, data1[j:j+2048]), axis=1)
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
#         x = np.zeros((2048, 0))
#         print(len(file_list))
#         print(x)
#         for i in range(len(file_list)):
#             file = scio.loadmat(os.path.join(data_dir, file_list[i]))
#             for k in file.keys():
#                 file_matched = re.match('data', k)
#                 if file_matched:
#                     key = file_matched.group()
#             data1 = np.array(file[key][0: 61440]) #0:80624
#             print(len(data1))
#             for j in range(0, len(data1)-2047, 2048):
#                   x = np.concatenate((x, data1[j:j+2048]), axis=1)
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
# data_dir_train = './data2/600'
# data_dir_test = './data2/800'
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
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import time
import torch
from torch.utils import data as da
from hhhh import DoubleChannelNet
from hhhh import DoubleChannelNet
from matplotlib import pyplot as plt
def plot_tsne(features, labels, colors, markers, sizes, title):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    colors: (list) 每个类别对应的颜色，长度为类别数
    markers: (list) 每个类别对应的形状，长度为类别数
    sizes: (list) 每个类别对应的大小，长度为类别数
    '''
    tsne = TSNE(perplexity=5, n_components=2, init='pca', random_state=0)
    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    latent = features
    tsne_features = tsne.fit_transform(latent)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)
    print('labels shape:', labels.shape)
    # 将对降维的特征进行可视化
    for i in range(class_num):
        idx = np.where(labels == i)
        plt.scatter(tsne_features[idx, 0], tsne_features[idx, 1], s=sizes[i], c=colors[i], marker=markers[i], label=i)
    # 设置点的颜色为标签，使用hsv颜色映射，设置点的形状为圆点
    # plt.xlim([-4, 4])
    # plt.ylim([-4, 4])
    plt.xlabel('x')  # 设置x轴名称
    plt.ylabel('y')  # 设置y轴名称
    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]
    rc = {'font.sans-serif': 'Times New Roman'}
    sns.set(font_scale=1, rc=rc, style='white')
    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                         palette=sns.color_palette(colors),
                         markers=markers,
                         sizes=sizes,
                         data=df, legend=True)
    fontsize1 = 20
    ax.set_xlabel('x', fontsize=fontsize1)  # 定义x轴标签和大小
    ax.set_ylabel('y', fontsize=fontsize1)  # 定义y轴标签和大小


if __name__ == '__main__':
    import time
    import torch
    from torch.utils import data as da
    from hhhh import DoubleChannelNet
    from hhhh import DoubleChannelNet
    from matplotlib import pyplot as plt
    from torchvision.models._utils import IntermediateLayerGetter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
markers = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
sizes = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
'''
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook


    def hook(model, input_, output):
        print("最后一层输出：", output.shape)


   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  hyj_model = DoubleChannelNet().to(device)
   device1=torch.device("cpu")
   print(hyj_model)
   hyj_model.load_state_dict(torch.load('weight/DoubleChannelNet1-OLSR-SGDP-bn-32-5.pt'))
   hyj_model.eval()
   hyj_model.fc.register_forward_hook(get_activation('hhh'))

    for i in train_loader:
        k1=i[0].to(device, dtype=torch.float32)

       hyj_model(k1)
       k = activation['hhh']
         k=k.to(device1)
       plt.figure(figsize=(12, 12))

        plt.imshow(k[0])
        plt.axis('off')
        plt.show()
'''


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


#
hyj_model = DoubleChannelNet().to(device)
hyj_model.load_state_dict(torch.load('weight/SEU.pt'))
hyj_model.eval()
activation = {}

# hyj_model.fu_3.register_forward_hook(get_activation('hhh'))
# hyj_model.fc.register_forward_hook(get_activation('hhh'))
hyj_model.conv1_ch1.register_forward_hook(get_activation('yyy'))
test_features = []
test_labels = []
i = 0
for img, label in test_loader:
    img = img.to(device, dtype=torch.float32)
    label = label.to(device, dtype=torch.long)
    out = hyj_model(img)
    # hyj_model.fc.register_forward_hook(get_activation('hhh'))
    bn = activation['yyy']
    # 获取指定层的输出
    # out = get_layer_output(model, img, layer_name)
    bn = torch.squeeze(bn).float()
    test_features.append(bn.cpu().detach().numpy())
    test_labels.append(label.cpu().detach().numpy())

test_features = np.concatenate(test_features, axis=0)
test_labels = np.concatenate(test_labels, axis=0)
tsne = TSNE(n_components=2, random_state=42)
test_features_2d = tsne.fit_transform(test_features.reshape(test_features.shape[0], -1))
print("test_features shape:", test_features.shape)
print("test_labels shape:", test_labels.shape)
print(test_labels)
# 调用 plot_tsne 函数并传入测试集的特征和标签
plot_tsne(test_features_2d, test_labels, colors, markers, sizes, title="t-SNE of raw layer")
time.sleep(3)
a = 666

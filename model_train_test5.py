
import torch.nn as nn
import numpy as np
# from datasave import train_loader, test_loader, X_test
from pyparsing import results

from early_stopping import EarlyStopping
from label_smoothing import OLSR,LSR
from oneD_Meta_ACON import MetaAconC
import time
import math
import torch
import torch.nn as n
from rbn import RepresentativeBatchNorm1d
from termcolor import cprint
import torch.nn.functional as F
from pytorch_lightning.utilities.seed import seed_everything
from torchsummary import summary
from adabn import reset_bn, fix_bn
from duibimodel import DoubleChannelNet,DCABiGRU,QCNN,EWSNET,MIXCNN,MA1DCNN,WDCNN,RNNWDCNN
from AdamP_amsgrad import AdamP, SGDP
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
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


#
setup_seed(150)

# seed_everything(0)

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x
class AlexNet(nn.Module):

    def __init__(self, in_channel=1, out_channel=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_channel),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6)
        x = self.classifier(x)
        return x
import scipy.signal
from torch.utils import data as da
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from statistics import stdev
num_repeats = 5
snr_values = [-4, -2, 0, 2, 4, 6, 8, 10, None]
# snr_values = [-4, -2]
raw_nums = [25, 50, 75, 100]
# raw_nums = [25, 50]
loads=['Dataset_A']
# loads=['Dataset_A', 'Dataset_B']
models=[MA1DCNN(10,1)]
# models=[MA1DCNN(10,1),WDCNN(),RNNWDCNN(),EWSNET(),QCNN(),AlexNet()]
# models=[DCABiGRU(),QCNN()]
all_results = []
for model in models:
    for load in loads:
        for raw_num in raw_nums:
            for snr in snr_values:
                print(f"Running experiment for model: {model.__class__.__name__} and load={load} and raw_num={raw_num} and SNR={snr} ")
                model_results = {
                    "model": model.__class__.__name__,
                    "load": load,
                    "raw_num": raw_num,
                    "snr": snr,
                    "train_accuracies": [],
                    "test_accuracies": []
                }
                for _ in range(num_repeats):


                    class Data(object):
                        def __init__(self):
                            self.data = self.get_data(load)
                            self.label = self.get_label()

                        def file_list(self, load):
                            return os.listdir(f'./data/{load}')

                        def get_data(self, load):
                            file_list = self.file_list(load)
                            x = np.zeros((1024, 0))
                            print(len(file_list))
                            print(x)
                            for i in range(len(file_list)):
                                file = scio.loadmat(f'./data/{load}/{file_list[i]}')
                                for k in file.keys():
                                    file_matched = re.match('X\d{3}_DE_time', k)
                                    if file_matched:
                                        key = file_matched.group()
                                data1 = np.array(file[key][0: 1024 * raw_num])  # 0:80624
                                print(len(data1))
                                for j in range(0, len(data1) - 1023, 1024):
                                    x = np.concatenate((x, data1[j:j + 1024]), axis=1)
                            return x.T

                        def get_label(self):
                            file_list = self.file_list(load)
                            title = np.array([i.replace('.mat', '') for i in file_list])
                            label = title[:, np.newaxis]
                            label_copy = np.copy(label)
                            for _ in range(raw_num - 1):
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
                    X_train, X_test, y_train, y_test = train_test_split(Data_with_noise, y, test_size=0.3, random_state=2,
                                                                        stratify=y)
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
                    # print(len(Train))
                    Test = TestDataset()
                    # print(Test)
                    # for data in Test:
                    #     txt, label = data
                    #     print(txt.shape, label)
                    train_loader = da.DataLoader(Train, batch_size=32, shuffle=True)
                    test_loader = da.DataLoader(Test, batch_size=10, shuffle=False)
                    print(f"test_loader batch_size: {test_loader.batch_size}")
                    print(f"test_loader length: {len(test_loader)}")
                    print(len(test_loader.dataset))
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    model = model.to(device)
                    summary(model, input_size=(1, 1024))
                    # criterion = LSR()
                    criterion = OLSR()
                    # criterion = nn.CrossEntropyLoss()
                    bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
                    others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
                    parameters = [{'parameters': bias_list, 'weight_decay': 0},
                                  {'parameters': others_list}]

                    optimizer = AdamP(model.parameters(), lr=0.001, weight_decay=0.0001, nesterov=True, amsgrad=True)

                    losses = []
                    acces = []
                    eval_losses = []
                    eval_acces = []
                    early_stopping = EarlyStopping(patience=10, verbose=True)
                    starttime = time.time()
                    for epoch in range(100):
                        train_loss = 0
                        train_acc = 0
                        model.train()
                        for img, label in train_loader:
                            img = img.float()
                            img = img.to(device)
                            # label = (np.argmax(label, axis=1)+1).reshape(-1, 1)
                            # label=label.float()

                            label = label.to(device)
                            label = label.long()
                            out = model(img)
                            out = torch.squeeze(out).float()
                            # label=torch.squeeze(label)

                            # out_1d = out.reshape(-1)
                            # label_1d = label.reshape(-1)

                            loss = criterion(out, label)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            # print(scheduler.get_lr())
                            train_loss += loss.item()

                            _, pred = out.max(1)
                            num_correct = (pred == label).sum().item()
                            acc = num_correct / img.shape[0]
                            train_acc += acc

                        losses.append(train_loss / len(train_loader))
                        acces.append(train_acc / len(train_loader))
                        #
                        eval_loss = 0
                        eval_acc = 0
                        model.eval()
                        model.apply(fix_bn)
                        with torch.no_grad():
                            for img, label in test_loader:
                                img = img.type(torch.FloatTensor)
                                img = img.to(device)
                                label = label.to(device)
                                label = label.long()
                                # img = img.view(img.size(0), -1)
                                out = model(img)
                                out = torch.squeeze(out).float()
                                loss = criterion(out, label)
                                #
                                eval_loss += loss.item()
                                #
                                _, pred = out.max(1)
                                num_correct = (pred == label).sum().item()
                                # print(pred, '\n\n', label)
                                acc = num_correct / img.shape[0]
                                eval_acc += acc
                        eval_losses.append(eval_loss / len(test_loader))
                        eval_acces.append(eval_acc / len(test_loader))
                        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
                              .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                                      eval_loss / len(test_loader), eval_acc / len(test_loader)))
                        # df = df.append({'epoch': epoch, 'train_loss': losses, 'train_acc': acces, 'test_loss': eval_losses,
                        #                 'test_acc': eval_acces}, ignore_index=True)

                    result_info = {
                        "model": model.__class__.__name__,
                        "load": load,
                        "raw_num": raw_num,
                        "snr": snr,
                        "train_losses": losses,
                        "train_accuracies": acces,
                        "test_losses": eval_losses,
                        "test_accuracies": eval_acces
                    }

                    all_results.append(result_info)
                    # df.to_excel('excel/training_results.xlsx', index=False)
                    endtime = time.time()
                    dtime = endtime - starttime
                    print("time：%.8s s" % dtime)
                    torch.save(model.state_dict(), 'weight/MCFRNN-CWRU-D.pt')

# 转换结果为 Pandas DataFrame
results_df = pd.DataFrame(all_results)

# 输出每种模型、每种 SNR、每种 raw 数和每种 load 的平均准确率和标准差
for model_name, group in results_df.groupby(['model', 'snr', 'raw_num', 'load']):
    avg_train_acc = np.mean(group['train_accuracies'])
    std_train_acc = stdev(group['train_accuracies'])
    avg_test_acc = np.mean(group['test_accuracies'])
    std_test_acc = stdev(group['test_accuracies'])

    print(f"Model: {model_name[0]}, SNR: {model_name[1]}, Raw Num: {model_name[2]}, Load: {model_name[3]}")
    print(f"Average Train Accuracy: {avg_train_acc:.4f} ± {std_train_acc:.4f}")
    print(f"Average Test Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")

# Save the DataFrame to a CSV file
results_df.to_csv('weight/experiment_results.csv', index=False)
# torch.save(model, 'weight/DoubleChannelNet1-OLSR-SGDP-bn-32-5.pt')
import pandas as pd
pd.set_option('display.max_columns', None)  #
pd.set_option('display.max_rows', None)  #
import matplotlib.pyplot as plt
# 绘制loss曲线
# plt.subplot(2, 1, 1)
# plt.plot(losses, label='train loss')
# plt.plot(eval_losses, label='eval loss')
# plt.legend(loc='best')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Train and Eval Loss')
#
# # 绘制accuracy曲线
# plt.subplot(2, 1, 2)
# plt.plot(acces, label='train acc')
# plt.plot(eval_acces, label='eval acc')
# plt.legend(loc='best')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Train and Eval Accuracy')

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from scipy.io import loadmat
import scipy.io as sio
plt.rcParams["font.sans-serif"]=["Times New Roman"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
# Define a function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Greens):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    # fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# def calculate_metrics(cm):
#     # Calculate metrics for each class
#     metrics = {}
#     for i in range(len(cm)):
#         tp = cm[i][i]
#         fp = np.sum(cm[:, i]) - tp
#         fn = np.sum(cm[i, :]) - tp
#         tn = np.sum(cm) - tp - fp - fn
#         accuracy = (tp + tn) / (tp + tn + fp + fn)
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#         f1_score = 2 * (precision * recall) / (precision + recall)
#         metrics[i] = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1_score
#         }
#     # Calculate overall metrics
#     tp_total = np.sum(np.diag(cm))
#     fp_total = np.sum(cm) - tp_total
#     fn_total = np.sum(np.sum(cm, axis=1)) - tp_total
#     recall_total = tp_total / (tp_total + fn_total)
#     precision_total = tp_total / (tp_total + fp_total)
#     f1_score_total = 2 * (precision_total * recall_total) / (precision_total + recall_total)
#     metrics['overall'] = {
#         'accuracy': np.trace(cm) / np.sum(cm),
#         'precision': precision_total,
#         'recall': recall_total,
#         'f1_score': f1_score_total
#     }
#     return metrics
# # Get the predicted labels for the test data
# model.eval()
# y_pred = []
# y_true = []
# # for data, label in test_loader:
# #     data = data.to(device)
# #     label = label.to(device)
# #     output = model(data)
# for img, label in test_loader:
#     img = img.type(torch.FloatTensor)
#     img = img.to(device)
#     label = label.to(device)
#     label = label.long()
#     # img = img.view(img.size(0), -1)
#     out = model(img)
#     out = torch.squeeze(out).float()
#     _, pred = out.max(1)
#     y_pred.extend(pred.cpu().numpy())
#     y_true.extend(label.cpu().numpy())
#
# # Compute the confusion matrix
# cm = confusion_matrix(y_true, y_pred)
#
# # Plot the confusion matrix
# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # Replace with your actual class labels
# plt.figure(figsize=(6, 4))
# # plot_confusion_matrix(cm, classes=class_names, normalize=True)
# plot_confusion_matrix(cm, classes=class_names, normalize=False)
# plt.show()
# metrics = calculate_metrics(cm)
#
# # Print metrics for each class
# for i, class_name in enumerate(class_names):
#     print(f"Metrics for class {class_name}:")
#     print(f"Accuracy: {metrics[i]['accuracy']:.3f}")
#     print(f"Precision: {metrics[i]['precision']:.3f}")
#     print(f"Recall: {metrics[i]['recall']:.3f}")
#     print(f"F1-score: {metrics[i]['f1_score']:.3f}\n")
# print("Overall metrics:")
# print(f"Accuracy: {metrics['overall']['accuracy']:.3f}")
# print(f"Precision: {metrics['overall']['precision']:.3f}")
# print(f"Recall: {metrics['overall']['recall']:.3f}")
# print(f"F1-score: {metrics['overall']['f1_score']:.3f}\n")
def calculate_metrics(cm):
    # Calculate metrics for each class
    metrics = {}
    for i in range(len(cm)):
        tp = cm[i][i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        metrics[i] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    # Calculate overall metrics
    overall_metrics = {}
    overall_metrics['accuracy'] = sum([metrics[i]['accuracy'] for i in range(len(metrics))]) / len(metrics)
    overall_metrics['precision'] = sum([metrics[i]['precision'] for i in range(len(metrics))]) / len(metrics)
    overall_metrics['recall'] = sum([metrics[i]['recall'] for i in range(len(metrics))]) / len(metrics)
    overall_metrics['f1_score'] = sum([metrics[i]['f1_score'] for i in range(len(metrics))]) / len(metrics)

    metrics['overall'] = overall_metrics

    return metrics


# Get the predicted labels for the test data
model.eval()
y_pred = []
y_true = []
for img, label in test_loader:
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    label = label.to(device)
    label = label.long()
    out = model(img)
    out = torch.squeeze(out).float()
    _, pred = out.max(1)
    y_pred.extend(pred.cpu().numpy())
    y_true.extend(label.cpu().numpy())

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Replace with your actual class labels
plt.figure(figsize=(6, 4))
# plot_confusion_matrix(cm, classes=class_names, normalize=True)
plot_confusion_matrix(cm, classes=class_names, normalize=False)
plt.show()

metrics = calculate_metrics(cm)

# Print metrics for each class and overall
# for i, class_name in enumerate(class_names):
#     print(f"Metrics for class {class_name}:")
#     print(f"Accuracy: {metrics[i]['accuracy']:.4f}")
#     print(f"Precision: {metrics[i]['precision']:.4f}")
#     print(f"Recall: {metrics[i]['recall']:.4f}")
#     print(f"F1-score: {metrics[i]['f1_score']:.4f}\n")

print(f"Overall metrics:")
print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
print(f"Precision: {metrics['overall']['precision']:.4f}")
print(f"Recall: {metrics['overall']['recall']:.4f}")
print(f"F1-score: {metrics['overall']['f1_score']:.4f}")



import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# # 设置散点形状
# maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# # 设置散点颜色
# colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
#           'hotpink']
# # 图例名称
# Label_Com = ['a', 'b', 'c', 'd']
# # 设置字体格式
# font1 = {'family': 'Times New Roman',
#          'weight': 'bold',
#          'size': 32,
#          }
# def plot_tsne(features, labels):
#     '''
#     features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
#     label:(N) 有N个标签
#     '''
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
#     latent = features
#     tsne_features = tsne.fit_transform(latent)  # 将特征使用PCA降维至2维
#     print('tsne_features的shape:', tsne_features.shape)
#     print('labels shape:', labels.shape)
#     # plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  # 将对降维的特征进行可视化
#     plt.scatter(tsne_features[:, 0], tsne_features[:, 1], s=10,c=labels, cmap='set3', marker='.')
#     # 设置点的颜色为标签，使用hsv颜色映射，设置点的形状为圆点
#     plt.legend()
#     plt.show()
#     plt.clf()
#     df = pd.DataFrame()
#     df["y"] = labels
#     df["comp-1"] = tsne_features[:, 0]
#     df["comp-2"] = tsne_features[:, 1]
#     rc = {'font.sans-serif':'Times New Roman'}
#     sns.set(font_scale=1,rc=rc,style='white')
#     # class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#     ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                     palette=sns.color_palette("hls", class_num),
#                     # hue_order=class_labels,
#                     data=df,legend=True,style =df.y.tolist())
#     fontsize1 = 13
#     ax.set_xlabel('x',fontsize=fontsize1)  # 定义x轴标签和大小
#     ax.set_ylabel('y',fontsize=fontsize1)  # 定义y轴标签和大小
#     x_ticks = np.arange(-5,6)
#     ax.set_xticklabels(x_ticks,fontsize =fontsize1) # 定义x轴坐标和大小
#     ax.set_yticklabels(x_ticks,fontsize=fontsize1)
    # plt.savefig('D:\qc\第二篇小论文\实验结果\tsne图\1.png') # 保存图片到本地
    # plt.show() # 显示图片
#绘制tsne
def plot_tsne(features, labels, colors, markers, sizes):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    colors: (list) 每个类别对应的颜色，长度为类别数
    markers: (list) 每个类别对应的形状，长度为类别数
    sizes: (list) 每个类别对应的大小，长度为类别数
    '''
    tsne = TSNE(perplexity=5,n_components=2, init='pca', random_state=0)
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
    # plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]
    rc = {'font.sans-serif':'Times New Roman'}
    sns.set(font_scale=1,rc=rc,style='white')
    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette(colors),
                    markers=markers,
                    sizes=sizes,
                    data=df,legend=True)
    fontsize1 = 20
    ax.set_xlabel('x',fontsize=fontsize1)  # 定义x轴标签和大小
    ax.set_ylabel('y',fontsize=fontsize1)  # 定义y轴标签和大小

    # x_ticks = np.arange(-0.5,0.5)
    # ax.set_xticklabels(x_ticks,fontsize =fontsize1) # 定义x轴坐标和大小
    # ax.set_yticklabels(x_ticks, fontsize=fontsize1)
# def plot_tsne(features, labels, colors, markers, sizes):
#     '''
#     features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
#     label:(N) 有N个标签
#     colors:(K) 颜色列表，其中K代表类别数目
#     markers:(K) 形状列表，其中K代表类别数目
#     sizes:(K) 点大小列表，其中K代表类别数目
#     '''
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
#     latent = features
#     tsne_features = tsne.fit_transform(latent)  # 将特征使用PCA降维至2维
#     print('tsne_features的shape:', tsne_features.shape)
#     print('labels shape:', labels.shape)
#
#     # 将对降维的特征进行可视化
#     for i in range(class_num):
#         idx = np.where(labels == i)
#         plt.scatter(tsne_features[idx, 0], tsne_features[idx, 1], s=sizes[i], c=colors[i], marker=markers[i], label=i)
#
#     plt.xlim(-4, 4)
#     plt.ylim(-4, 4)
#     plt.legend()
#     plt.show()
#     plt.clf()
#     df = pd.DataFrame()
#     df["y"] = labels
#     df["comp-1"] = tsne_features[:, 0]
#     df["comp-2"] = tsne_features[:, 1]
#     rc = {'font.sans-serif': 'Times New Roman'}
#     sns.set(font_scale=1, rc=rc, style='white')
#     # class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#     ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                          palette=sns.color_palette("hls", class_num),
#                          # hue_order=class_labels,
#                          data=df, legend=True, style=df.y.tolist())
#     fontsize1 = 13
#     ax.set_xlabel('x', fontsize=fontsize1)  # 定义x轴标签和大小
#     ax.set_ylabel('y', fontsize=fontsize1)  # 定义y轴标签和大小
#     x_ticks = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
#     ax.set_xticklabels(x_ticks, fontsize=fontsize1)  # 定义x轴坐标和大小
#     ax.set_yticklabels(x_ticks, fontsize=fontsize1)
#     ax.set_xlim(-4, 4)
#     ax.set_ylim(-4, 4)
#     ax.legend()
#     plt.show()

colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
markers = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
sizes = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
# layer_name = 'layer1'
test_features = []
test_labels = []
model.eval()
# 在训练过程中获取测试集的特征和标签
for img, label in test_loader:
    img = img.to(device, dtype=torch.float32)
    label = label.to(device, dtype=torch.long)
    out = model(img)
    # 获取指定层的输出
    #out = get_layer_output(model, img, layer_name)
    out = torch.squeeze(out).float()
    test_features.append(out.cpu().detach().numpy())
    test_labels.append(label.cpu().detach().numpy())
test_features = np.concatenate(test_features, axis=0)
test_labels = np.concatenate(test_labels, axis=0)
print("test_features shape:", test_features.shape)
print("test_labels shape:", test_labels.shape)
print(test_labels)
# 调用 plot_tsne 函数并传入测试集的特征和标签
# plot_tsne(test_features, test_labels, colors, markers, sizes,title="t-SNE of final layer")
plot_tsne(test_features, test_labels, colors, markers, sizes)


#绘制roc图
from sklearn.metrics import roc_curve, auc
# Get the predicted scores for the test data
model.eval()
y_scores = []
y_true = []
for img, label in test_loader:
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    label = label.to(device)
    label = label.long()
    out = model(img)
    out = torch.squeeze(out).float().detach()  # 分离出out
    y_scores.append(out.cpu().numpy())
    y_true.append(label.cpu().numpy())
y_scores = np.concatenate(y_scores, axis=0)
y_true = np.concatenate(y_true, axis=0)
# Compute the ROC curves and AUC values for each class
fpr = {}
tpr = {}
roc_auc = {}
n_classes = 10
class_names = ['Normal','Ball-0.007', 'Ball-0.014','Ball-0.021','IR-0.007','IR-0.014','IR-0.021','OR-0.007','OR-0.014','OR-0.021']
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true, y_scores[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
# Plot the ROC curves for each class
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(class_names):
    plt.plot(fpr[i], tpr[i], label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
# Add diagonal line representing random guess
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# Set plot title and labels
plt.title('ROC Curves for each class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Add legend
plt.legend(loc="lower right")
# Show the plot
plt.show()
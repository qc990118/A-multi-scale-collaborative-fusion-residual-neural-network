import itertools
import math
import os

import torch
import torch.nn as nn
import numpy as np


from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.svm._libsvm import predict
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn import Linear, BatchNorm1d, ReLU, Sigmoid, AvgPool1d, Conv1d, Softmax
from torch.utils.data import DataLoader
from rbn import RepresentativeBatchNorm1d
import datasave

from datasave import train_loader, test_loader, X_test

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
setup_seed(20)

# seed_everything(0)

# class swish(nn.Module):
#     def __init__(self):
#         super(swish, self).__init__()
#
#     def forward(self, x):
#         x = x * F.sigmoid(x)
#         return x
# def reset_bn(module):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         module.track_running_stats = False
# def fix_bn(module):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         module.track_running_stats = True

# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)


# class Bottle2neck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
#         """ Constructor
#         Args:
#             inplanes: input channel dimensionality
#             planes: output channel dimensionality
#             stride: conv stride. Replaces pooling layer.
#             downsample: None when stride = 1
#             baseWidth: basic width of conv3x3
#             scale: number of scale.
#             type: 'normal': normal set. 'stage': first block of a new stage.
#         """
#         super(Bottle2neck, self).__init__()
#
#         width = int(math.floor(planes * (baseWidth / 64.0)))
#         self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(width * scale)
#
#         if scale == 1:
#             self.nums = 1
#         else:
#             self.nums = scale - 1
#         if stype == 'stage':
#             self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
#         convs = []
#         bns = []
#         for i in range(self.nums):
#             if i == 0:
#                 convs.append(nn.Conv1d(width, width, kernel_size=1, bias=False))
#                 bns.append(nn.BatchNorm1d(width))
#                 convs.append(nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
#                 bns.append(nn.BatchNorm1d(width))
#                 convs.append(nn.Conv1d(width, width, kernel_size=5, stride=stride, padding=2, bias=False))
#                 bns.append(nn.BatchNorm1d(width))
#             # else:
#             #     convs.append(nn.Conv1d(width * 3, width, kernel_size=1, bias=False))
#             #     convs.append(nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
#             #     convs.append(nn.Conv1d(width, width, kernel_size=5, stride=stride, padding=2, bias=False))
#             #     bns.append(nn.BatchNorm1d(width))
#             #     bns.append(nn.BatchNorm1d(width))
#             #     bns.append(nn.BatchNorm1d(width))
#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList(bns)
#
#         self.conv3 = nn.Conv1d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm1d(planes * self.expansion)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stype = stype
#         self.scale = scale
#         self.width = width
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         spx = torch.split(out, self.width, 1)
#         for i in range(self.nums):
#             if i == 0 or self.stype == 'stage':
#                 sp = spx[i]
#             else:
#                 sp = sp + spx[i]
#             sp = self.convs[i * 3](sp)
#             sp = self.relu(self.bns[i * 3](sp))
#             sp = self.convs[i * 3 + 1](sp)
#             sp = self.relu(self.bns[i * 3 + 1](sp))
#             sp = self.convs[i * 3 + 2](sp)
#             sp = self.relu(self.bns[i * 3 + 2](sp))
#             if i == 0:
#                 out = sp
#             else:
#                 out = torch.cat((out, sp), 1)
#         if self.scale != 1 and self.stype == 'normal':
#             out = torch.cat((out, spx[self.nums]), 1)
#         elif self.scale != 1 and self.stype == 'stage':
#             out = torch.cat((out, self.pool(spx[self.nums])), 1)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # self.pool_w = nn.AdaptiveAvgPool1d(1)
        self.pool_w = nn.AdaptiveMaxPool1d(1)
        mip = max(6, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip, track_running_stats=False)
        self.act = MetaAconC(mip)
        self.conv_w = nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, w = x.size()
        x_w = self.pool_w(x)
        y = torch.cat([identity, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_ww, x_c = torch.split(y, [w, 1], dim=2)
        a_w = self.conv_w(x_ww)
        a_w = a_w.sigmoid()
        out = identity * a_w
        return out
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth #到底啥叫baseWidth呢
        self.scale = scale
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.MetaAconC(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
class DepthwiseConv1D(nn.Module):
    def __init__(self, dim_in, kernel_size, dilation_rate, depth_multiplier, padding="same",
                                       use_bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim_in, out_channels=dim_in * depth_multiplier, kernel_size=kernel_size, stride=1, padding=padding, groups=dim_in,
                              bias=use_bias, dilation=dilation_rate)

    def forward(self, x):
        x = self.conv(x)
        return x

class Mixconv(nn.Module):
    def __init__(self, channal=64, kersize=64, m=1, c=1, dim_in=128):
        super(Mixconv, self).__init__()
        self.depth_conv_1 = DepthwiseConv1D(dim_in=dim_in, kernel_size=kersize, dilation_rate=m, depth_multiplier=c, padding="same",
                                       use_bias=False)
        self.act_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm1d(dim_in * m)
        self.conv_1 = nn.Conv1d(dim_in * m, channal, kernel_size=1, stride=1, padding="same")
        self.act_3 = nn.ReLU()
        self.bn_3 = nn.BatchNorm1d(channal)

    def forward(self, x):
        x1 = x
        x = self.depth_conv_1(x)
        x = self.act_2(x)
        x = self.bn_2(x)
        x = torch.add(x, x1)
        x = self.conv_1(x)
        x = self.act_3(x)
        x = self.bn_3(x)
        return x




######################################################################################################################
class MIXCNN(nn.Module):
    def __init__(self):
        super(MIXCNN, self).__init__()
        self.conv_1 = nn.Conv1d(1, 128, kernel_size=32, stride=4)
        self.bn_1 = nn.BatchNorm1d(128)
        self.act_1 = nn.ReLU()
        self.mix_1 = Mixconv(dim_in=128, channal=128, kersize=64, m=1, c=1)
        self.mix_2 = Mixconv(dim_in=128, channal=128, kersize=64, m=1, c=1)
        self.mix_3 = Mixconv(dim_in=128, channal=128, kersize=64, m=1, c=1)
        self.bn_2 = nn.BatchNorm1d(128)
        self.act_2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv_1(x)
        x = F.pad(x, (387, 388), "constant", 0)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.mix_1(x)
        x = self.mix_2(x)
        x = self.mix_3(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        x = self.pool(x).squeeze()
        x = self.fc(x)
        return x


class CIM0(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CIM0, self).__init__()
        #定义激活函数
        # act_fn = nn.ReLU(inplace=True)
        act_fn = MetaAconC(out_dim)
        #定义两个1D卷积层，输入通道数和输出通道数都为out_dim，kernel_size=3, stride=1, padding=1
        self.layer_10 = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        # 定义两个1D卷积层接BatchNorm1d和ReLU激活函数的组合
        self.layer_11 = nn.Sequential(nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(out_dim), act_fn, )
        self.layer_21 = nn.Sequential(nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(out_dim), act_fn, )
        # 定义可训练参数gamma1和gamma2，初始值均为0
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        # 定义一个包含1个1D卷积层、BatchNorm1d和ReLU激活函数的组合的顺序结构
        self.layer_ful1 = nn.Sequential(nn.Conv1d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm1d(out_dim), act_fn, )

    def forward(self, x1, x2):
        ################################

        x_rgb = self.layer_10(x1)
        x_dep = self.layer_20(x2)

        rgb_w = nn.Sigmoid()(x_rgb)
        dep_w = nn.Sigmoid()(x_dep)

        # 计算x1和x_dep的逐元素积
        x_rgb_w = x1.mul(dep_w)
        x_dep_w = x2.mul(rgb_w)
        # 计算x_rgb的残差值
        x_rgb_r = x_rgb_w + x1
        x_dep_r = x_dep_w + x2

        ## fusion
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)


        ful_out = torch.cat((x_rgb_r, x_dep_r), dim=1)

        out1 = self.layer_ful1(ful_out)

        return out1


class CIM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CIM, self).__init__()

        act_fn = nn.ReLU(inplace=True)
        self.reduc_1 = nn.Sequential(nn.Conv1d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_2 = nn.Sequential(nn.Conv1d(in_dim, out_dim, kernel_size=1), act_fn)

        self.layer_10 = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_11 = nn.Sequential(nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(out_dim), act_fn, )
        self.layer_21 = nn.Sequential(nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(out_dim), act_fn, )

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.layer_ful1 = nn.Sequential(nn.Conv1d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm1d(out_dim), act_fn, )
        self.layer_ful2 = nn.Sequential(nn.Conv1d(out_dim + out_dim // 2, out_dim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm1d(out_dim), act_fn, )

    def forward(self, x1, x2, xx):
        ################################
        x_rgb = self.reduc_1(x1)
        x_dep = self.reduc_2(x2)

        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)

        rgb_w = nn.Sigmoid()(x_rgb1)
        dep_w = nn.Sigmoid()(x_dep1)

        ##
        x_rgb_w = x_rgb.mul(dep_w)
        x_dep_w = x_dep.mul(rgb_w)

        x_rgb_r = x_rgb_w + x_rgb
        x_dep_r = x_dep_w + x_dep

        ## fusion
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)


        ful_out = torch.cat((x_rgb_r, x_dep_r), dim=1)

        out1 = self.layer_ful1(ful_out)
        out2 = self.layer_ful2(torch.cat([out1, xx], dim=1))

        return out2


class MFA(nn.Module):
    def __init__(self, in_dim):
        super(MFA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.layer_10 = nn.Conv1d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv1d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.layer_cat1 = nn.Sequential(nn.Conv1d(in_dim * 2, in_dim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(in_dim), )

    def forward(self, x_ful, x1, x2):
        ################################

        x_ful_1 = x_ful.mul(x1)
        x_ful_2 = x_ful.mul(x2)

        x_ful_w = self.layer_cat1(torch.cat([x_ful_1, x_ful_2], dim=1))
        out = self.relu(x_ful + x_ful_w)

        return out

class BasicConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = MetaAconC(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MFAM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MFAM, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_1_1 = BasicConv1d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv1d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv1d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv1d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv1d(out_channels, out_channels, 3, stride=1, padding=1)

        self.conv_3_1 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_5_1 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        ###+
        x1 = self.conv_1_1(x)
        x2 = self.conv_1_2(x)
        x3 = self.conv_1_3(x)

        x_3_1 = self.relu(self.conv_3_1(x2))  ## (BS, 32, ***, ***)
        x_5_1 = self.relu(self.conv_5_1(x3))  ## (BS, 32, ***, ***)

        x_3_2 = self.relu(self.conv_3_2(x_3_1 + x_5_1))  ## (BS, 64, ***, ***)
        x_5_2 = self.relu(self.conv_5_2(x_5_1 + x_3_1))  ## (BS, 64, ***, ***)

        x_mul = torch.mul(x_3_2, x_5_2)

        out = self.relu(x1 + self.conv_1_5(x_mul + x_3_1 + x_5_1))

        return out
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1),

            BasicConv1d(out_channel, out_channel, kernel_size=3, padding=1),
            BasicConv1d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1),

            BasicConv1d(out_channel, out_channel, kernel_size=5, padding=2),
            BasicConv1d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv1d(in_channel, out_channel, 1),

            BasicConv1d(out_channel, out_channel, kernel_size=7, padding=3),
            BasicConv1d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv1d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv1d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
import torch
import torch.nn as nn
def maxpool():
    pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    return pool


class DCABiGRU(nn.Module):
    def __init__(self):
        super(DCABiGRU, self).__init__()
        self.p1_1 = nn.Sequential(nn.Conv1d(1, 50, kernel_size=18, stride=2),
                                  nn.BatchNorm1d(50),
                                  MetaAconC(50))
        self.p1_2 = nn.Sequential(nn.Conv1d(50, 30, kernel_size=10, stride=2),
                                  nn.BatchNorm1d(30),
                                  MetaAconC(30))
        self.p1_3 = nn.MaxPool1d(2, 2)
        self.p2_1 = nn.Sequential(nn.Conv1d(1, 50, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(50),
                                  MetaAconC(50))
        self.p2_2 = nn.Sequential(nn.Conv1d(50, 40, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(40),
                                  MetaAconC(40))
        self.p2_3 = nn.MaxPool1d(2, 2)
        self.p2_4 = nn.Sequential(nn.Conv1d(40, 30, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(30),
                                  MetaAconC(30))
        self.p2_5 = nn.Sequential(nn.Conv1d(30, 30, kernel_size=6, stride=2),
                                  nn.BatchNorm1d(30),
                                  MetaAconC(30))
        self.p2_6 = nn.MaxPool1d(2, 2)
        self.p3_0 = CoordAtt(30, 30)
        self.p3_1 = nn.Sequential(nn.GRU(124, 64, bidirectional=True))  #
        # self.p3_2 = nn.Sequential(nn.LSTM(128, 512))
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.p4 = nn.Sequential(nn.Linear(30, 10))

    def forward(self, x):
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        encode = torch.mul(p1, p2)
        # p3 = self.p3_2(self.p3_1(encode))
        p3_0 = self.p3_0(encode).permute(1, 0, 2)
        p3_2, _ = self.p3_1(p3_0)
        # p3_2, _ = self.p3_2(p3_1)
        p3_11 = p3_2.permute(1, 0, 2)  #
        p3_12 = self.p3_3(p3_11).squeeze()
        # p3_11 = h1.permute(1,0,2)
        # p3 = self.p3(encode)
        # p3 = p3.squeeze()
        # p4 = self.p4(p3_11)  # LSTM(seq_len, batch, input_size)
        # p4 = self.p4(encode)
        p4 = self.p4(p3_12)
        return p4
import torch
import torch.nn as nn

from torchsummary import summary
import torch.nn.functional as F
from ConvQuadraticOperation import ConvQuadraticOperation

class QCNN(nn.Module):
    """
    QCNN builder
    """

    def __init__(self, ) -> object:
        super(QCNN, self).__init__()
        self.cnn = nn.Sequential()
        # self.cnn1 = nn.Sequential()
        self.cnn.add_module('Conv1D_1', ConvQuadraticOperation(1, 16, 64, 8, 28))
        self.cnn.add_module('BN_1', nn.BatchNorm1d(16))
        self.cnn.add_module('Relu_1', nn.ReLU())
        self.cnn.add_module('MAXPool_1', nn.MaxPool1d(2, 2))
        self.__make_layerq(16, 32, 1, 2)
        self.__make_layerq(32, 64, 1, 3)
        self.__make_layerq(64, 64, 1, 4)
        self.__make_layerq(64, 64, 1, 5)
        self.__make_layerq(64, 64, 0, 6)

        self.fc1 = nn.Linear(64, 100)
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)



    def __make_layerq(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), ConvQuadraticOperation(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        # self.cnn.add_module('DP_%d' %(nb_patch), nn.Dropout(0.5))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def __make_layerc(self, in_channels, out_channels, padding, nb_patch):
        self.cnn1.add_module('Conv1D_%d' % (nb_patch), nn.Conv1d(in_channels, out_channels, 3, 1, padding))
        self.cnn1.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        # self.cnn.add_module('DP_%d' %(nb_patch), nn.Dropout(0.5))
        self.cnn1.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn1.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def forward(self, x):
        out1 = self.cnn(x)
        out = self.fc1(out1.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
def cal_auc_roc(y_true, y_pred_score, num_class=7):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    y_true_one_hot = label_binarize(y_true, classes=list(range(num_class)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    return roc_auc, fpr, tpr

#%%
# def conv1d_blk(x,filters,kernel_size,strides,padding):
#     x1 = layers.Conv1D(filters, kernel_size, strides, padding)(x)
#     x2 = layers.BatchNormalization()(x1)
#     x3 = layers.Activation('relu')(x2)
#     x3 = layers.Dropout(0.2)(x3)
#
#     return x3


def feature_map_reshape12(x):
    '''
    - Description: Covert the 1D feature map into 2D feature map
    - Input: x, shape:B*L*C, x is batch of 1D feature maps with C channels. - - Note that the length L should be 2^k, where k is positive integer
    - Output:y, shape: B*H*W*C

    Example:
    input_shape = (11, 512, 7)
    x = tf.random.normal(input_shape)
    output_shape: TensorShape([11, 16, 32, 7])
    '''

    x_shape = x.shape
    len_feature_map1d = x_shape[2]

    tmp = np.log2(len_feature_map1d)
    if int(tmp % 2) == 0:
        h = int(2**(tmp/2))
        w = int(len_feature_map1d/h)
    elif int(tmp % 2) == 1:
        h = int(2**((tmp-1)/2))
        w = int(len_feature_map1d/h)


    samples_1d = x
    feature_map_shape2d = [-1, x_shape[1], h, w]
    # feature_map_shape2d = [-1, x_shape[2], w, h] # there is no significant difference between the h-first and w-first

    samples_2d = torch.reshape(samples_1d, feature_map_shape2d)
    y = samples_2d

    return y


def gram_blk(x):
    '''
    - Description: Calculate the Gram Enhanced feature map based on 2D feature map
    - Input: x, shape: B*H*W*C
    - Output: y = (x*x.T)*x*(x.T*x)
    '''
    xg1 = torch.einsum('bcij,bcjk->bcik', torch.permute(x, (0, 1, 3, 2)), x)
    xg2 = torch.einsum('bcij,bcjk->bcik', x, torch.permute(x, (0, 1, 3, 2)))
    y = torch.einsum('bcij,bcjk->bcik', xg2, x)
    y = torch.einsum('bcij,bcjk->bcik', y, xg1)
    return y


def feature_map_reshape21(x):
    '''
    - Description: Covert feature maps (generated by gram_blk) from 2D to 1D
    - Input: x, shape:B*H*W*C, x is batch of 2D feature maps with C channels.
    - Output:y, shape: B*L*C
    - Note that the length L should be 2^k, where k is positive integer


    Example:
    input_shape = [11, 16, 32, 7]
    x = tf.random.normal(input_shape)
    output_shape: TensorShape([11, 512, 7])
    '''

    x_shape = x.shape
    len_feature_map1d = x_shape[2]*x_shape[3]
    feature_map_shape2d = [-1,  x_shape[1], len_feature_map1d]
    y = torch.reshape(x, feature_map_shape2d)

    return y

def batch_gram_blk(x):

    x = feature_map_reshape12(x)
    x = gram_blk(x)
    x = feature_map_reshape21(x)
    # y = layers.LayerNormalization(axis=1)(x3)
    y = F.layer_norm(x, normalized_shape=(x.size()[1], x.size()[2]))
    return y

def batch_fft_blk(x):
    data_length = x.shape[2]
    # x1 = x
    x = torch.abs(torch.fft.rfft(x)) / data_length
    # x3 = x2
    # x4 = layers.LayerNormalization(axis=1)(x3)
    x = F.layer_norm(x, normalized_shape=(x.size()[1], x.size()[2]))
    return x


class conv1d_blk(nn.Module):

    def __init__(self, in_channel, filters, kernel_size, strides, padding):
        super(conv1d_blk, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, filters, kernel_size, strides, padding)
        self.bn1 = nn.BatchNorm1d(filters)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        return x


# def the_blk(xr0, xg0, xf0, ):
#     '''
#     disconnect the Fourier bridges, only Gramian bridge is kept.
#     '''
#     xr1 = conv1d_blk(xr0, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
#     xr1_g = batch_gram_blk(xr1)
#     xg0_c = conv1d_blk(xg0, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
#     xg1 = torch.concat([xr1_g, xg0_c], 2)
#     # xg1_0   = tf.concat([xr1_g,xg0_c], axis = 2)
#     # xg1 = conv1d_blk(xg1_0, filters = filters, kernel_size = 1, strides = 1, padding = padding) #notice that
#     xf1 = conv1d_blk(xf0, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
#
#
#     return xr1, xg1, xf1

class the_blk(nn.Module):

    def __init__(self, in_channel, filters=16, kernel_size=3, strides=1, padding=0):
        super(the_blk, self).__init__()
        self.conv1d_blk1 = conv1d_blk(in_channel, filters=filters, kernel_size=kernel_size, strides=strides,
                                      padding=padding)
        if in_channel == 1:
            self.conv1d_blk2 = conv1d_blk(in_channel=1, filters=filters, kernel_size=kernel_size, strides=strides,
                                          padding=padding)
            self.conv1d_blk3 = conv1d_blk(in_channel=2, filters=filters, kernel_size=kernel_size, strides=strides,
                                          padding=padding)
        else:
            self.conv1d_blk2 = conv1d_blk(2 * in_channel, filters=filters, kernel_size=kernel_size, strides=strides,
                                          padding=padding)
            self.conv1d_blk3 = conv1d_blk(in_channel, filters=filters, kernel_size=kernel_size, strides=strides,
                                          padding=padding)
    def forward(self, xr0, xg0, xf0):
        xr1 = self.conv1d_blk1(xr0)
        xr1_g = batch_gram_blk(xr1)
        xg0_c = self.conv1d_blk2(xg0)
        xg1 = torch.concat([xr1_g, xg0_c], 1)
        # xg1_0   = tf.concat([xr1_g,xg0_c], axis = 2)
        # xg1 = conv1d_blk(xg1_0, filters = filters, kernel_size = 1, strides = 1, padding = padding) #notice that
        xf1 = self.conv1d_blk3(xf0)
        return xr1, xg1, xf1

class the_model(nn.Module):

    def __init__(self, class_number=10):
        super(the_model, self).__init__()
        self.the_blk1 = the_blk(in_channel=1, filters=32, kernel_size=31, strides=1, padding=15)
        self.the_blk2 = the_blk(in_channel=32, filters=32, kernel_size=31, strides=2, padding=15)
        self.the_blk3 = the_blk(in_channel=32, filters=64, kernel_size=15, strides=2, padding=7)
        self.the_blk4 = the_blk(in_channel=64, filters=64, kernel_size=15, strides=2, padding=7)
        self.the_blk5 = the_blk(in_channel=64, filters=128, kernel_size=5, strides=2, padding=2)
        self.fc = nn.Linear(512, class_number)

    def forward(self, in_x):
        xr0 = in_x
        xg0 = batch_gram_blk(in_x)
        xg0_f = batch_fft_blk(xg0)
        xr0_f = batch_fft_blk(xr0)
        xf0 = torch.concat([xg0_f, xr0_f], 1)
        xr1, xg1, xf1 = self.the_blk1(xr0, xg0, xf0)
        xr1, xg1, xf1 = self.the_blk2(xr1, xg1, xf1)
        xr1, xg1, xf1 = self.the_blk3(xr1, xg1, xf1)
        xr1, xg1, xf1 = self.the_blk4(xr1, xg1, xf1)
        xr2, xg2, xf2 = self.the_blk5(xr1, xg1, xf1)
        x21 = F.adaptive_avg_pool1d(xr2, 1)
        x22 = F.adaptive_avg_pool1d(xg2, 1)
        x23 = F.adaptive_avg_pool1d(xf2, 1)
        x2 = torch.concat([x21, x22, x23], 1).squeeze()
        x2 = self.fc(x2)
        return x2
import torch
import torch.nn as nn
from my_utils.Attention_Block import Flatten, SELayer1
# from data_generate_utils.my_utils import Euclidean_Distance, vis_tSNE, plot_confusion_matrix, t_sne
# from Attention_Block import Flatten, SELayer2
from my_utils.semi_utils import assign_cluster, update_cluster, compute_logits
from my_utils.plot_utils import plot_confusion_matrix, umap_fun2, t_sne


def conv_block(in_channels, out_channels):
    """
    w_out = [w_in + 2*padding - dilation*(k_size-1)]/stride + 1
    if dilation=1, stride=1, w_out = w_in + 2*padding - k_size
    to keep the w_in and w_out the same, padding=(k_size-1)/2
    :param in_channels:
    :param out_channels:
    :return:
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),  # [N, 64,2048]
        nn.BatchNorm1d(out_channels, track_running_stats=True),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
    )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.channel = 1
        self.conv1 = conv_block(self.channel, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.se1 = SELayer1(64)
        # self.se13 = SELayer1(64)
        # self.se14 = SELayer1(64)
        # self.se2 = SELayer2(64)

    def forward(self, x):
        net = self.conv1(x)
        # net = self.se1(net)
        net = self.conv2(net)
        # net = self.se1(net)
        net = self.conv3(net)
        net = self.se1(net)
        net = self.conv4(net)
        net = self.se1(net)
        net = Flatten()(net)
        return net


class BasicModel(nn.Module):
    def __init__(self, img_w):
        super(BasicModel, self).__init__()
        self.encoder = Encoder()
        self.img_w = img_w
        self.chn = 1
        # self.num_cluster_steps = 1  # 1~5
        # self.num_cluster_steps = 1  # 1~5
        self.num_cluster_steps = 3  # 3
        print('---The num_cluster_steps is: %d ---\n' % self.num_cluster_steps)

    def get_encoded_inputs(self, *x_list):
        """Runs the reference and candidate data through the feature model Encoder.
        Returns:
          h_s: [B, N, D]
          h_u: [B, P, D]
          h_q: [B, M, D]
        """
        bsize = x_list[0].shape[0]
        num = [xx.shape[1] for xx in x_list]
        x_all = torch.cat(x_list, dim=1)
        x_all = x_all.view(-1, self.chn, self.img_w)  # [B*N, 1, 2048]
        h_all = self.encoder(x_all)
        h_all = h_all.view(bsize, sum(num), -1)
        h_list = torch.split(h_all, num, dim=1)
        return h_list

    def compute_protos(self, h_s):
        """Computes the prototypes, cluster centers.
        Args:
          h_s: [K or B, N, D], Train features.其中B即为n_way or K
        Returns:
          p: [K, D], Test prediction.
        """
        p = h_s.mean(dim=1)
        # p = h_s.sum(dim=1)
        return p

    def kmeans_predict(self, n_class, y_s, x_s, x_u, x_q, vis, cm):
        """
        x_s: [B, Ns, 2048, 1]
        x_u: [B, Nu, 2048, 1]
        x_q: [B, Nq, 2048, 1]
        y_s: [B, Ns, 1]
        :returns: logits [B, N, K]
        """
        h_s, h_u, h_q = self.get_encoded_inputs(x_s, x_u, x_q)  # 3个均为 [B, N, D]
        y_s = y_s.squeeze(dim=-1)
        # if cm:
        #     data = h_q.detach().reshape(-1, h_q.shape[-1]).cpu().numpy()  # (nc*nq, z_dim)
        #     label = y_s.cpu().numpy().reshape(-1)  # (n,)
        #     vis_tSNE(data, label, classes=n_class, vis=vis, name='query after', n_dim=2)
        #     t_sne(data, label, classes=n_class, name='query after', n_dim=2)
        #     # [nc, z_dim]==>[nc, 1, z_dim]==>[nc, ns, z_dim]
        # 使用.expand似乎会造成存储空间不连续，类似于对tensor实施转置也会造成存储空间不连续
        # 此时，就不能使用.view()，解决方法是在.view()前增加.contiguous()

        protos = self.compute_protos(h_s)  # [K, D] 即 [Nc, D]

        # Hard assignment for training images.
        prob_s = [None] * n_class
        for kk in range(n_class):
            prob_s[kk] = torch.unsqueeze(torch.eq(y_s, kk).float(), dim=2)  # [B, N, 1]
        prob_s = torch.cat(prob_s, dim=2)  # [B, N, K] the probability of support_data

        h_all = torch.cat([h_s, h_u], dim=1)  # [B, Ns+Nu, D]

        # Run clustering.
        for tt in range(self.num_cluster_steps):
            prob_u = assign_cluster(protos, h_u)  # [B, P, K]
            prob_all = torch.cat([prob_s, prob_u], dim=1)  # [B, N+P, K]
            prob_all = torch.detach(prob_all)
            # ##################################问题所在############################
            # protos = protos + update_cluster(h_all, prob_all)  # [K, D]
            protos = update_cluster(h_all, prob_all)  # [K, D]

        logits = compute_logits(protos, h_q)  # [B, N, K]
        return logits, h_q


class ssProto(BasicModel):
    """
    train or test the Model
    """

    def __init__(self, img_w, device):
        super(ssProto, self).__init__(img_w=img_w)
        self.d = device

    def forward(self, nc, x_s, x_u, x_q, vis, cm=False):
        bsize = x_s.shape[0]
        nq, ns = x_q.shape[1], x_s.shape[1]
        assert nc == bsize

        y_q = torch.arange(0, nc, device=self.d,
                           dtype=torch.long).reshape(nc, 1, 1).repeat(1, nq, 1)  # [C, N, 1]
        y_s = torch.arange(0, nc, device=self.d,
                           dtype=torch.long).reshape(nc, 1, 1).repeat(1, ns, 1)
        # y_s = torch.arange(0, nc, device=self.d).reshape(nc, 1, 1).expand(nc, ns, 1).long()
        logits, h_q = self.kmeans_predict(nc, y_s, x_s, x_u, x_q, vis, cm=cm)  # [B, N, K]
        # log_p_y = torch.log_softmax(logits, dim=-1).view(bsize, nq, -1)  # [B, M, K]
        log_p_y = torch.log_softmax(logits, dim=-1)  # [B, M, K]
        loss = -log_p_y.gather(dim=2, index=y_q).squeeze().view(-1).mean()  # [B, M, 1]==>[B*M]
        y_hat = log_p_y.max(dim=2)[1]  # [B, M]
        acc = torch.eq(y_hat, y_q.squeeze(dim=-1)).float().mean()
        if cm and ns > 5:
            y_pre = y_hat.view(-1).cpu().detach().numpy()
            y_true = y_q.reshape(-1).squeeze().cpu().detach().numpy()
            # t_sne(input_data=h_q.reshape(nc * ns, -1).cpu().numpy(), input_label=y_true, classes=nc)
            # plot_confusion_matrix(y_true=y_true, y_pred=y_pre, disp_acc=True)

        return loss, {'loss': loss.item(), 'acc': acc.item()}, \
            h_q.reshape(nc * ns, -1).cpu()
import torch
import torch.nn as nn
# from wmodelsii8 import Sin_fast as fast
# from wmodelsii3 import Laplace_fast as fast
from thresholds import Shrinkagev2 as fast
# from wsinc import SincConv_fast as fast
from thresholds import Shrinkagev3ppp2 as sage

class Mish1(nn.Module):
    def __init__(self):
        super(Mish1, self).__init__()
        self.mish = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.mish(x)


class EWSNET(nn.Module):
    def __init__(self):
        super(EWSNET, self).__init__()    #85,42,70   #63,31,75
        self.p1_0 = nn.Sequential(  # nn.Conv1d(1, 50, kernel_size=18, stride=2),
            # fast(out_channels=64, kernel_size=250, stride=1),
            # fast1(out_channels=70, kernel_size=84, stride=1),
            nn.Conv1d(1, 64, kernel_size=250, stride=1, bias=True),
            nn.BatchNorm1d(64),
            Mish1()
        )
        self.p1_1 = nn.Sequential(nn.Conv1d(64, 16, kernel_size=18, stride=2, bias=True),
                                  # fast(out_channels=50, kernel_size=18, stride=2),
                                  nn.BatchNorm1d(16),
                                  Mish1()
                                  )
        self.p1_2 = nn.Sequential(nn.Conv1d(16, 10, kernel_size=10, stride=2, bias=True),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                  )
        self.p1_3 = nn.MaxPool1d(kernel_size=2)
        self.p2_1 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=6, stride=1, bias=True),
                                  # fast(out_channels=50, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(32),
                                  Mish1()
                                  )
        self.p2_2 = nn.Sequential(nn.Conv1d(32, 16, kernel_size=6, stride=1, bias=True),
                                  nn.BatchNorm1d(16),
                                  Mish1()
                                  )
        self.p2_3 = nn.MaxPool1d(kernel_size=2)
        self.p2_4 = nn.Sequential(nn.Conv1d(16, 10, kernel_size=6, stride=1, bias=True),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                  )
        self.p2_5 = nn.Sequential(nn.Conv1d(10, 10, kernel_size=8, stride=2, bias=True),
                                  # nn.Conv1d(10, 10, kernel_size=6, stride=2),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                 )  # PRelu
        self.p2_6 = nn.MaxPool1d(kernel_size=2)
        self.p3_0 = sage(channel=64, gap_size=1)
        self.p3_1 = nn.Sequential(nn.Conv1d(64, 10, kernel_size=43, stride=4, bias=True),
                                  nn.BatchNorm1d(10),
                                  Mish1()
                                 )
        self.p3_2 = nn.MaxPool1d(kernel_size=2)
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.p4 = nn.Sequential(nn.Linear(10, 10))




    def forward(self, x):
        x = self.p1_0(x)
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        x = self.p3_2(self.p3_1(x + self.p3_0(x)))
        x = torch.add(x, torch.add(p1, p2))
        x = self.p3_3(x).squeeze()
        x = self.p4(x)
        return x

class Conv1dSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv1dSamePadding, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.width = in_channels

        self.padding = self.calculate_padding()
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size,
                                    stride=stride)

    def calculate_padding(self):
        """
        W/S = (W-K+TP)/S+1    # new W bothers with stride

        # solve for TP (total padding)
        W/S-1 = (W-K+TP)/S
        S(W/S-1) = W-K+TP
        TP = S(W/S-1)-W+K

        TP = W-S-W+K
        TP = K-S
        """
        # p = (self.kernel_size // 2 - 1) * self.stride + 1
        # p = (self.stride * (self.width / self.stride - 1) - self.width + self.kernel_size) / 2
        total_padding = max(0, self.kernel_size - self.stride)
        p1 = total_padding // 2
        p2 = total_padding - p1
        return p1, p2

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv_layer(x)

class CAM(nn.Module):
    def __init__(self, num_filters):
        super(CAM, self).__init__()
        self.num_filters = num_filters
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(self.num_filters, self.num_filters // 2, 1, padding="same")
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(self.num_filters // 2, self.num_filters, 1, padding="same")
        self.batchnorm = nn.BatchNorm1d(self.num_filters)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b1 = self.avgpool(x)
        b1 = self.conv1(b1)
        b1 = self.relu(b1)

        b1 = self.conv2(b1)
        b1 = self.batchnorm(b1)
        b1 = self.sigmoid(b1)

        b2 = torch.multiply(x, b1)
        out = x + b2

        return out

class EAM(nn.Module):
    def __init__(self, num_filters, kernel_size):
        super(EAM, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(num_filters, 1, 1)
        self.batchnorm = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()

        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.conv1(x)
        b1 = self.batchnorm(b1)
        b1 = self.sigmoid(b1)

        b2 = self.conv2(x)
        b2 = self.relu(b2)
        b3 = torch.multiply(b1, b2)
        o = x + b3

        return o

class MA1DCNN(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(MA1DCNN, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels, 32, 32, padding="same")
        self.relu1 = nn.ReLU()
        self.eam1 = EAM(32, 32)
        self.cam1 = CAM(32)

        self.conv2 = Conv1dSamePadding(32, 32, 16, stride=2)
        self.relu2 = nn.ReLU()
        self.eam2 = EAM(32, 16)
        self.cam2 = CAM(32)

        self.conv3 = Conv1dSamePadding(32, 64, 9, stride=2)
        self.relu3 = nn.ReLU()
        self.eam3 = EAM(64, 9)
        self.cam3 = CAM(64)

        self.conv4 = Conv1dSamePadding(64, 64, 6, stride=2)
        self.relu4 = nn.ReLU()
        self.eam4 = EAM(64, 6)
        self.cam4 = CAM(64)

        self.conv5 = Conv1dSamePadding(64, 128, 3, stride=4)
        self.relu5 = nn.ReLU()
        self.eam5 = EAM(128, 3)
        self.cam5 = CAM(128)

        self.conv6 = Conv1dSamePadding(128, 128, 3, stride=4)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.eam1(x)
        x = self.cam1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.eam2(x)
        x = self.cam2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.eam3(x)
        x = self.cam3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.eam4(x)
        x = self.cam4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.eam5(x)
        x = self.cam5(x)

        x = self.conv6(x)
        # x = torch.permute(x, (0, 2, 1))
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)




class DoubleChannelNet(nn.Module):
    def __init__(self):
        super(DoubleChannelNet, self).__init__()
        # 定义第一个通道的卷积层
        self.conv1_ch1 = nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=1)
        self.bn1_ch1 = nn.BatchNorm1d(64)
        self.act_1 = MetaAconC(64)
        # self.res_ch1_1=Res2Net(Bottle2neck, [1, 1, 1, 1],)
        self.bottle2neck_ch1_0 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_0=CoordAtt(64,64)
        self.gcm_ch1_0 = GCM(64,64)
        # self.mfam_ch1_0 = MFAM(64, 64)
        self.bottle2neck_ch1_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_1 = CoordAtt(64, 64)
        # self.res_ch1_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch1_1 = MFA(4)
        self.gcm_ch1_1 = GCM(64, 64)
        # self.mfam_ch1_1 = MFAM(64, 64)
        self.bottle2neck_ch1_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_2 = CoordAtt(64, 64)
        self.gcm_ch1_2 = GCM(64, 64)
        # self.bottle2neck_ch1_3 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.gcm_ch1_3 = GCM(64, 64)
        # self.mfam_ch1_2 = MFAM(64, 64)
        # self.res_ch1_3 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch1_2 = MFA(4)

        # 定义第二个通道的卷积层
        self.conv1_ch2 = nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=1)
        self.bn1_ch2 = nn.BatchNorm1d(64)
        self.act_2 =MetaAconC(64)
        # self.res_ch2_1 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        self.bottle2neck_ch2_0 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch2_0 = CoordAtt(64, 64)
        self.gcm_ch2_0 = GCM(64,64)
        # self.mfam_ch2_0 = MFAM(64, 64)
        self.bottle2neck_ch2_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch2_1= CoordAtt(64, 64)
        self.gcm_ch2_1 = GCM(64, 64)
        # self.mfam_ch2_1 = MFAM(64, 64)
        # self.res_ch2_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch2_1 = MFA(4)
        self.bottle2neck_ch2_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch2_2 = CoordAtt(64, 64)
        self.gcm_ch2_2 = GCM(64, 64)
        # self.bottle2neck_ch2_3 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.gcm_ch2_3 = GCM(64, 64)
        # self.mfam_ch2_2 = MFAM(64, 64)
        # self.res_ch2_3 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch2_2 = MFA(4)
        # 定义共享的卷积层

        self.fu_0 = CIM0(64,64)  # MixedFusion_Block_IMfusion

        self.fu_1 = CIM(64, 128)

        self.gcm_0 = GCM(128,64)
        # self.mfam_0 = MFAM(128, 64)

        self.fu_2 = CIM(64, 128)
        self.gcm_1 = GCM(128,64)
        # self.mfam_1 = MFAM(128, 64)

        self.fu_3 = CIM(64, 128)
        self.gcm_2 = GCM(128,64)
        # self.fu_4 = CIM(64, 128)
        # self.gcm_3 = GCM(128,64)
        # self.mfam_2 = MFAM(128, 64)
        self.bn_2 = nn.BatchNorm1d(192)
        # self.bn_2 = nn.RepresentativeBatchNorm1d(192)
        self.act_3 = nn.ReLU()
        # self.at1=CoordAtt(192,192)
        # 定义全连接层
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):

        # 对第一个通道进行卷积和池化操作
        x1 = self.conv1_ch1(x)
        x1 = self.bn1_ch1(x1)
        x1=self.act_1(x1)


        # 对第二个通道进行卷积和池化操作
        x2 = self.conv1_ch2(x)
        x2 = self.bn1_ch2(x2)
        x2 = self.act_2(x2)

        # 将pool1_ch1和pool1_ch2拼接
        fu_0 = self.fu_0(x1,x2)
        # print('fu_0 shape:', fu_0.shape)
        x1 =self.bottle2neck_ch1_0(x1)
        x1=self.CoordAtt1_ch1_0(x1)
        # print('x1 shape:', x1.shape)
        x1=self.gcm_ch1_0(x1)
        # print('x1-1 shape:', x1.shape)
        # x1 = self.res_ch1_1(x1)

        x2 =self.bottle2neck_ch2_0(x2)
        x2 = self.CoordAtt1_ch2_0(x2)
        # print('x2 shape:', x2.shape)
        x2=self.gcm_ch2_0(x2)
        # print('x2-2 shape:', x2.shape)
        # x2 = self.res_ch2_1(x2)

        fu_1 = self.fu_1(x1, x2,fu_0)
        # print('fu_1 shape:', fu_1.shape)
        # pool_fu_1 = self.pool_fu_1(fu_1)
        # print('pool_fu_1 shape:', pool_fu_1     .shape)
        gcm_0=self.gcm_0(fu_1)
        # print('gcm_0 shape:', gcm_0.shape)
        x1 =self.bottle2neck_ch1_1(x1)
        x1 = self.CoordAtt1_ch1_1(x1)
        # print('x1-2 shape:', x1.shape)
        # x1 = self.res_ch1_2(x1)
        x1=self.gcm_ch1_1(x1)
        # print('x1-3 shape:', x1.shape)
        x2 =self.bottle2neck_ch2_1(x2)
        x2 = self.CoordAtt1_ch2_1(x2)
        # print('x2-2 shape:', x2.shape)
        x2=self.gcm_ch2_1(x2)
        # print('x2-3 shape:', x2.shape)
        fu_2 = self.fu_1(x1, x2,gcm_0)
        # print('fu_2 shape:', fu_2.shape)
        gcm_1 = self.gcm_1(fu_2)
        # print('gcm_1 shape:', gcm_1.shape)
        x1 = self.bottle2neck_ch1_2(x1)
        x1 = self.CoordAtt1_ch1_2(x1)
        x1=self.gcm_ch1_2(x1)
        # print('x1 shape:', x1.shape)
        # x1 = self.mfa_ch1_2(x1)
        x2 = self.bottle2neck_ch2_2(x2)
        x2 = self.CoordAtt1_ch2_2(x2)
        x2 = self.gcm_ch2_2(x2)
        # print('x2 shape:', x2.shape)
        # x2 = self.mfa_ch2_2(x2)
        fu_3 = self.fu_1(x1, x2, gcm_1)
        # print('fu_3 shape:', fu_3.shape)
        gcm_2 = self.gcm_2(fu_3)

        # x1 = self.bottle2neck_ch1_3(x1)
        # x1=self.gcm_ch1_3(x1)
        # # print('x1 shape:', x1.shape)
        # # x1 = self.mfa_ch1_2(x1)
        # x2 = self.bottle2neck_ch2_3(x2)
        # x2 = self.gcm_ch2_3(x2)
        # fu_4 = self.fu_1(x1, x2, gcm_2)
        # # print('fu_3 shape:', fu_3.shape)
        # gcm_3 = self.gcm_2(fu_4)
        # print('gcm_2 shape:', gcm_2.shape)
        x=torch.cat([x1,x2,gcm_2],dim=1)
        # print('x shape:', x.shape)
        x=self.bn_2(x)
        x=self.act_3(x)
        # x=self.at1(x)
        x=self.pool(x).squeeze()
        # print('x shape:', x.shape)


        # 将最后一层的输出通过全连接层得到最终的输出
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
class WDCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(WDCNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64,stride=16,padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2,stride=2)
            )


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # nn.AdaptiveMaxPool1d(4)
        )  # 32, 12,12     (24-2) /2 +1

        self.fc=nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_channel)
        )


    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x) #[16 64]
        # print(x.shape)
        x = self.layer2(x)  #[32 124]
        # print(x.shape)
        x = self.layer3(x)#[64 61]
        # print(x.shape)
        x = self.layer4(x)#[64 29]
        # print(x.shape)
        x = self.layer5(x)#[64 13]
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = Res2Net(Bottle2neck,[1,2,3,4]).to(device)
# model = DoubleChannelNet().to(device)
model = WDCNN().to(device)


# input1 = torch.randn(1, 1024)
# input2 = torch.randn(1, 1024)
# model = DoubleChannelNet(input_size=1024)
# output = model(input1, input2)
# print(output)
# x1 = torch.randn(1, 1024)
# x2 = torch.randn(1, 1024)
# output = model(x1, x2)
# x1 = torch.randn(1, 1024).to(device)
# x2 = torch.randn(1, 1024).to(device)
# output = model.forward(x1, x2)

summary(model, input_size=(1,1024))
# def summarize_layer(model, layer_name):
#     layer = None
#     for name, module in model.named_modules():
#         if name == layer_name:
#             layer = module
#             break
#     if layer is None:
#         raise ValueError('Layer not found: ' + layer_name)
# summarize_layer(model, 'conv5')
# criterion = nn.CrossEntropyLoss()
# criterion = LSR()
# criterion = OLSR()
# criterion =nn.CrossEntropyLoss()
# criterion = OnlineLabelSmoothingLoss(smoothing=0.1)
# criterion = CrossEntropyLoss_LSR(device)
# from adabound import AdaBound
# optimizer = AdaBound(model.parameters(), lr=0.001, weight_decay=0.0001, amsbound=True)
# from EAdam import EAdam
# optimizer = EAdam(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True)
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.000, weight_decay=0.0001)
bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},
              {'parameters': others_list}]
# optimizer = Nadam(model.parameters())
# optimizer = RAdam(model.parameters())
# from torch_optimizer import AdamP
# from adamp import AdamP

from AdamP_amsgrad import AdamP,SGDP
# optimizer = AdamP(model.parameters(), lr=0.001, weight_decay=0.0001, nesterov=True, amsgrad=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150,
#                                                                      eta_min=1e-8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = LabelSmoothingCrossEntropy()

# optimizer = SGDP(model.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)
# from adabelief_pytorch import AdaBelief
# optimizer = AdaBelief(model.parameters(), lr=0.001, weight_decay=0.0001, weight_decouple=True)
# from ranger_adabelief import RangerAdaBelief
# optimizer = RangerAdaBelief(model.parameters(), lr=0.001, weight_decay=0.0001, weight_decouple=True)
snr_values = [-4, -2, 0, 2, 4, 6, 8, 10, None]
# snr_values = [0, 2, 4, 6, 8, 10, None]
experiments = 5
# experiment_results = []  # 存储每次实验的结果
# results_df = pd.DataFrame(columns=['SNR', 'Experiment', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])
raw_num = 100
for snr in snr_values:
    for experiment in range(experiments):
        print(f"Running experiment {experiment + 1} for SNR={snr}")


        class Data(object):
            def __init__(self):
                self.data = self.get_data()
                self.label = self.get_label()

            def file_list(self):
                return os.listdir('./data/Dataset_A')

            def get_data(self):
                file_list = self.file_list()
                x = np.zeros((1024, 0))
                # print(len(file_list))
                # print(x)
                for i in range(len(file_list)):
                    file = scio.loadmat('./data/Dataset_A/{}'.format(file_list[i]))
                    for k in file.keys():
                        file_matched = re.match('X\d{3}_DE_time', k)
                        if file_matched:
                            key = file_matched.group()
                    data1 = np.array(file[key][0: 102400])  # 0:80624
                    # print(len(data1))
                    for j in range(0, len(data1) - 1023, 1024):
                        x = np.concatenate((x, data1[j:j + 1024]), axis=1)
                return x.T

            def get_label(self):
                file_list = self.file_list()
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
        # 在这里开始你的数据处理和模型训练代码
        Data = Data()
        data = Data.data
        label = Data.label
        y = label.astype("int32")
        Data_with_noise = []
        for i in range(len(data)):
            noisy_data = add_gaussian_noise(data[i], snr=snr)
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
        # # X_train = torch.from_numpy(X_train).unsqueeze(-1)
        # # X_test = torch.from_numpy(X_test).unsqueeze(-1)
        #
        # X_train = torch.from_numpy(X_train)
        # X_test = torch.from_numpy(X_test)


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
        losses = []
        acces = []
        eval_losses = []
        eval_acces = []
        early_stopping = EarlyStopping(patience=10, verbose=True)
        starttime = time.time()
        for epoch in range(150):
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
            model.apply(reset_bn)
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
            early_stopping(eval_loss / len(test_loader), model)
            model.apply(fix_bn)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        endtime = time.time()
        dtime = endtime - starttime
        print("time：%.8s s" % dtime)
        torch.save(model.state_dict(), 'weight/DoubleChannelNet1-100--4.pt')

# losses = []
# acces = []
# eval_losses = []
# eval_acces = []
# early_stopping = EarlyStopping(patience=10, verbose=True)
# starttime = time.time()
# for epoch in range(150):
#     train_loss = 0
#     train_acc = 0
#     model.train()
#     for img, label in train_loader:
#         img = img.float()
#         img = img.to(device)
#         # label = (np.argmax(label, axis=1)+1).reshape(-1, 1)
#         # label=label.float()
#
#         label = label.to(device)
#         label = label.long()
#         out = model(img)
#         out = torch.squeeze(out).float()
#         # label=torch.squeeze(label)
#
#         # out_1d = out.reshape(-1)
#         # label_1d = label.reshape(-1)
#
#         loss = criterion(out, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print(scheduler.get_lr())
#         train_loss += loss.item()
#
#         _, pred = out.max(1)
#         num_correct = (pred == label).sum().item()
#         acc = num_correct / img.shape[0]
#         train_acc += acc
#
#     losses.append(train_loss / len(train_loader))
#     acces.append(train_acc / len(train_loader))
#     #
#     eval_loss = 0
#     eval_acc = 0
#     model.eval()
#     model.apply(reset_bn)
#     for img, label in test_loader:
#         img = img.type(torch.FloatTensor)
#         img = img.to(device)
#         label = label.to(device)
#         label = label.long()
#         # img = img.view(img.size(0), -1)
#         out = model(img)
#         out = torch.squeeze(out).float()
#         loss = criterion(out, label)
#         #
#         eval_loss += loss.item()
#         #
#         _, pred = out.max(1)
#         num_correct = (pred == label).sum().item()
#         # print(pred, '\n\n', label)
#         acc = num_correct / img.shape[0]
#         eval_acc += acc
#     eval_losses.append(eval_loss / len(test_loader))
#     eval_acces.append(eval_acc / len(test_loader))
#     print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
#           .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
#                   eval_loss / len(test_loader), eval_acc / len(test_loader)))
#     early_stopping(eval_loss / len(test_loader), model)
#     model.apply(fix_bn)
#     if early_stopping.early_stop:
#         print("Early stopping")
#         break
# endtime = time.time()
# dtime = endtime - starttime
# print("time：%.8s s" % dtime)
# torch.save(model.state_dict(), 'weight/DoubleChannelNet1-100--4.pt')
# torch.save(model, 'weight/DoubleChannelNet1-OLSR-SGDP-bn-32-5.pt')
import pandas as pd
pd.set_option('display.max_columns', None)  #
pd.set_option('display.max_rows', None)  #
import matplotlib.pyplot as plt
# 绘制loss曲线
plt.subplot(2, 1, 1)
plt.plot(losses, label='train loss')
plt.plot(eval_losses, label='eval loss')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Eval Loss')

# 绘制accuracy曲线
plt.subplot(2, 1, 2)
plt.plot(acces, label='train acc')
plt.plot(eval_acces, label='eval acc')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Eval Accuracy')

# 显示图像
plt.show()

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from scipy.io import loadmat
import scipy.io as sio
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
# Define a function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Reds):
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
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
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
    return metrics
# Get the predicted labels for the test data
model.eval()
y_pred = []
y_true = []
# for data, label in test_loader:
#     data = data.to(device)
#     label = label.to(device)
#     output = model(data)
for img, label in test_loader:
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    label = label.to(device)
    label = label.long()
    # img = img.view(img.size(0), -1)
    out = model(img)
    out = torch.squeeze(out).float()
    _, pred = out.max(1)
    y_pred.extend(pred.cpu().numpy())
    y_true.extend(label.cpu().numpy())

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # Replace with your actual class labels
plt.figure(figsize=(6, 4))
# plot_confusion_matrix(cm, classes=class_names, normalize=True)
plot_confusion_matrix(cm, classes=class_names, normalize=False)
plt.show()
metrics = calculate_metrics(cm)

# Print metrics for each class
for i, class_name in enumerate(class_names):
    print(f"Metrics for class {class_name}:")
    print(f"Accuracy: {metrics[i]['accuracy']:.3f}")
    print(f"Precision: {metrics[i]['precision']:.3f}")
    print(f"Recall: {metrics[i]['recall']:.3f}")
    print(f"F1-score: {metrics[i]['f1_score']:.3f}\n")
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

# 图例名称
# Label_Com = ['a', 'b', 'c', 'd']


test_features = []
test_labels = []
model.eval()
# 在训练过程中获取测试集的特征和标签

for img, label in test_loader:
    img = img.to(device, dtype=torch.float32)
    label = label.to(device, dtype=torch.long)
    out = model(img)
    out = torch.squeeze(out).float()
    test_features.append(out.cpu().detach().numpy())
    test_labels.append(label.cpu().detach().numpy())
test_features = np.concatenate(test_features, axis=0)
test_labels = np.concatenate(test_labels, axis=0)
print("test_features shape:", test_features.shape)
print("test_labels shape:", test_labels.shape)
print(test_labels)

# 调用 plot_tsne 函数并传入测试集的特征和标签
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
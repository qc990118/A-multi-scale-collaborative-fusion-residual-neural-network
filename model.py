import torch
import torch.nn as nn
import numpy as np


from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.svm._libsvm import predict
from timm.optim import AdaBelief
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
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


#
setup_seed(20)

# seed_everything(0)

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x
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

bottle_neck = Bottle2neck(inplanes=64, planes=64, stride=2, baseWidth=26, scale=4, stype='normal')


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


class MCFRNNNGSM(nn.Module):
    def __init__(self):
        super(MCFRNNNGSM, self).__init__()
        # 定义第一个通道的卷积层
        self.conv1_ch1 = nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=1)
        self.bn1_ch1 = nn.BatchNorm1d(64)
        self.act_1 = MetaAconC(64)
        # self.res_ch1_1=Res2Net(Bottle2neck, [1, 1, 1, 1],)
        self.bottle2neck_ch1_0 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_0=CoordAtt(64,64)
        # self.gcm_ch1_0 = GCM(64,64)
        # self.mfam_ch1_0 = MFAM(64, 64)
        self.bottle2neck_ch1_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_1 = CoordAtt(64, 64)
        # self.res_ch1_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch1_1 = MFA(4)
        # self.gcm_ch1_1 = GCM(64, 64)
        # self.mfam_ch1_1 = MFAM(64, 64)
        self.bottle2neck_ch1_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_2 = CoordAtt(64, 64)
        # self.gcm_ch1_2 = GCM(64, 64)
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
        # self.gcm_ch2_0 = GCM(64,64)
        # self.mfam_ch2_0 = MFAM(64, 64)
        self.bottle2neck_ch2_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch2_1= CoordAtt(64, 64)
        # self.gcm_ch2_1 = GCM(64, 64)
        # self.mfam_ch2_1 = MFAM(64, 64)
        # self.res_ch2_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch2_1 = MFA(4)
        self.bottle2neck_ch2_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch2_2 = CoordAtt(64, 64)
        # self.gcm_ch2_2 = GCM(64, 64)
        # self.bottle2neck_ch2_3 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.gcm_ch2_3 = GCM(64, 64)
        # self.mfam_ch2_2 = MFAM(64, 64)
        # self.res_ch2_3 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch2_2 = MFA(4)
        # 定义共享的卷积层

        self.fu_0 = CIM0(64,64)  # MixedFusion_Block_IMfusion

        # self.fu_1 = CIM(64, 128)
        #
        # # self.gcm_0 = GCM(128,64)
        # # self.mfam_0 = MFAM(128, 64)
        #
        # self.fu_2 = CIM(64, 128)
        # # self.gcm_1 = GCM(128,64)
        # # self.mfam_1 = MFAM(128, 64)
        #
        # self.fu_3 = CIM(64, 128)
        # # self.gcm_2 = GCM(128,64)
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
        # x1=self.gcm_ch1_0(x1)
        # print('x1-1 shape:', x1.shape)
        # x1 = self.res_ch1_1(x1)

        x2 =self.bottle2neck_ch2_0(x2)
        x2 = self.CoordAtt1_ch2_0(x2)
        # print('x2 shape:', x2.shape)
        # x2=self.gcm_ch2_0(x2)
        # print('x2-2 shape:', x2.shape)
        # x2 = self.res_ch2_1(x2)

        # fu_1 = self.fu_1(x1, x2,fu_0)
        # print('fu_1 shape:', fu_1.shape)
        # pool_fu_1 = self.pool_fu_1(fu_1)
        # print('pool_fu_1 shape:', pool_fu_1     .shape)
        # gcm_0=self.gcm_0(fu_1)
        # print('gcm_0 shape:', gcm_0.shape)
        x1 =self.bottle2neck_ch1_1(x1)
        x1 = self.CoordAtt1_ch1_1(x1)
        # print('x1-2 shape:', x1.shape)
        # x1 = self.res_ch1_2(x1)
        # x1=self.gcm_ch1_1(x1)
        # print('x1-3 shape:', x1.shape)
        x2 =self.bottle2neck_ch2_1(x2)
        x2 = self.CoordAtt1_ch2_1(x2)
        # print('x2-2 shape:', x2.shape)
        # x2=self.gcm_ch2_1(x2)
        # print('x2-3 shape:', x2.shape)
        # fu_2 = self.fu_1(x1, x2,gcm_0)
        # print('fu_2 shape:', fu_2.shape)
        # gcm_1 = self.gcm_1(fu_2)
        # print('gcm_1 shape:', gcm_1.shape)
        x1 = self.bottle2neck_ch1_2(x1)
        x1 = self.CoordAtt1_ch1_2(x1)
        # x1=self.gcm_ch1_2(x1)
        # print('x1 shape:', x1.shape)
        # x1 = self.mfa_ch1_2(x1)
        x2 = self.bottle2neck_ch2_2(x2)
        x2 = self.CoordAtt1_ch2_2(x2)
        # x2 = self.gcm_ch2_2(x2)
        # print('x2 shape:', x2.shape)
        # x2 = self.mfa_ch2_2(x2)
        # fu_3 = self.fu_1(x1, x2, gcm_1)
        # print('fu_3 shape:', fu_3.shape)
        # gcm_2 = self.gcm_2(fu_3)

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
        x=torch.cat([x1,x2,fu_0],dim=1)
        # print('x shape:', x.shape)
        x=self.bn_2(x)
        x=self.act_3(x)
        # x=self.at1(x)
        x=self.pool(x).squeeze()
        # print('x shape:', x.shape)9989


        # 将最后一层的输出通过全连接层得到最终的输出
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class MCFRNNNMRM(nn.Module):
    def __init__(self):
        super(MCFRNNNMRM, self).__init__()
        # 定义第一个通道的卷积层
        self.conv1_ch1 = nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=1)
        self.bn1_ch1 = nn.BatchNorm1d(64)
        self.act_1 = MetaAconC(64)
        # self.res_ch1_1=Res2Net(Bottle2neck, [1, 1, 1, 1],)
        # self.bottle2neck_ch1_0 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_0=CoordAtt(64,64)
        self.gcm_ch1_0 = GCM(64,64)
        # self.mfam_ch1_0 = MFAM(64, 64)
        # self.bottle2neck_ch1_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_1 = CoordAtt(64, 64)
        # self.res_ch1_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch1_1 = MFA(4)
        self.gcm_ch1_1 = GCM(64, 64)
        # self.mfam_ch1_1 = MFAM(64, 64)
        # self.bottle2neck_ch1_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
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
        # self.bottle2neck_ch2_0 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch2_0 = CoordAtt(64, 64)
        self.gcm_ch2_0 = GCM(64,64)
        # self.mfam_ch2_0 = MFAM(64, 64)
        # self.bottle2neck_ch2_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch2_1= CoordAtt(64, 64)
        self.gcm_ch2_1 = GCM(64, 64)
        # self.mfam_ch2_1 = MFAM(64, 64)
        # self.res_ch2_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch2_1 = MFA(4)
        # self.bottle2neck_ch2_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
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
        self.fc = nn.Linear(192, 5)

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
        # x1 =self.bottle2neck_ch1_0(x1)
        x1=self.CoordAtt1_ch1_0(x1)
        # print('x1 shape:', x1.shape)
        x1=self.gcm_ch1_0(x1)
        # print('x1-1 shape:', x1.shape)
        # x1 = self.res_ch1_1(x1)

        # x2 =self.bottle2neck_ch2_0(x2)
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
        # x1 =self.bottle2neck_ch1_1(x1)
        x1 = self.CoordAtt1_ch1_1(x1)
        # print('x1-2 shape:', x1.shape)
        # x1 = self.res_ch1_2(x1)
        x1=self.gcm_ch1_1(x1)
        # print('x1-3 shape:', x1.shape)
        # x2 =self.bottle2neck_ch2_1(x2)
        x2 = self.CoordAtt1_ch2_1(x2)
        # print('x2-2 shape:', x2.shape)
        x2=self.gcm_ch2_1(x2)
        # print('x2-3 shape:', x2.shape)
        fu_2 = self.fu_1(x1, x2,gcm_0)
        # print('fu_2 shape:', fu_2.shape)
        gcm_1 = self.gcm_1(fu_2)
        # print('gcm_1 shape:', gcm_1.shape)
        # x1 = self.bottle2neck_ch1_2(x1)
        x1 = self.CoordAtt1_ch1_2(x1)
        x1=self.gcm_ch1_2(x1)
        # print('x1 shape:', x1.shape)
        # x1 = self.mfa_ch1_2(x1)
        # x2 = self.bottle2neck_ch2_2(x2)
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
        # print('x shape:', x.shape)9989


        # 将最后一层的输出通过全连接层得到最终的输出
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
class MCFRNNNSDM(nn.Module):
    def __init__(self):
        super(MCFRNNNSDM, self).__init__()
        # 定义第一个通道的卷积层
        self.conv1_ch1 = nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=1)
        self.bn1_ch1 = nn.BatchNorm1d(64)
        self.act_1 = MetaAconC(64)
        # self.res_ch1_1=Res2Net(Bottle2neck, [1, 1, 1, 1],)
        self.bottle2neck_ch1_0 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.CoordAtt1_ch1_0 = CoordAtt(64, 64)
        self.gcm_ch1_0 = GCM(64, 64)
        # self.mfam_ch1_0 = MFAM(64, 64)
        self.bottle2neck_ch1_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.CoordAtt1_ch1_1 = CoordAtt(64, 64)
        # self.res_ch1_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch1_1 = MFA(4)
        self.gcm_ch1_1 = GCM(64, 64)
        # self.mfam_ch1_1 = MFAM(64, 64)
        self.bottle2neck_ch1_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.CoordAtt1_ch1_2 = CoordAtt(64, 64)
        self.gcm_ch1_2 = GCM(64, 64)
        # self.bottle2neck_ch1_3 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.gcm_ch1_3 = GCM(64, 64)
        # self.mfam_ch1_2 = MFAM(64, 64)
        # self.res_ch1_3 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch1_2 = MFA(4)

        # 定义第二个通道的卷积层
        self.conv1_ch2 = nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=1)
        self.bn1_ch2 = nn.BatchNorm1d(64)
        self.act_2 = MetaAconC(64)
        # self.res_ch2_1 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        self.bottle2neck_ch2_0 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.CoordAtt1_ch2_0 = CoordAtt(64, 64)
        self.gcm_ch2_0 = GCM(64, 64)
        # self.mfam_ch2_0 = MFAM(64, 64)
        self.bottle2neck_ch2_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.CoordAtt1_ch2_1 = CoordAtt(64, 64)
        self.gcm_ch2_1 = GCM(64, 64)
        # self.mfam_ch2_1 = MFAM(64, 64)
        # self.res_ch2_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch2_1 = MFA(4)
        self.bottle2neck_ch2_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.CoordAtt1_ch2_2 = CoordAtt(64, 64)
        self.gcm_ch2_2 = GCM(64, 64)
        # self.bottle2neck_ch2_3 = Bottle2neck(inplanes=64, planes=16, stride=1)
        # self.gcm_ch2_3 = GCM(64, 64)
        # self.mfam_ch2_2 = MFAM(64, 64)
        # self.res_ch2_3 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch2_2 = MFA(4)
        # 定义共享的卷积层

        self.fu_0 = CIM0(64, 64)  # MixedFusion_Block_IMfusion

        self.fu_1 = CIM(64, 128)

        self.gcm_0 = GCM(128, 64)
        # self.mfam_0 = MFAM(128, 64)

        self.fu_2 = CIM(64, 128)
        self.gcm_1 = GCM(128, 64)
        # self.mfam_1 = MFAM(128, 64)

        self.fu_3 = CIM(64, 128)
        self.gcm_2 = GCM(128, 64)
        # self.fu_4 = CIM(64, 128)
        # self.gcm_3 = GCM(128,64)
        # self.mfam_2 = MFAM(128, 64)
        self.bn_2 = nn.BatchNorm1d(192)
        # self.bn_2 = nn.RepresentativeBatchNorm1d(192)
        self.act_3 = nn.ReLU()
        # self.at1=CoordAtt(192,192)
        # 定义全连接层
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(192, 5)

    def forward(self, x):
        # 对第一个通道进行卷积和池化操作
        x1 = self.conv1_ch1(x)
        x1 = self.bn1_ch1(x1)
        x1 = self.act_1(x1)

        # 对第二个通道进行卷积和池化操作
        x2 = self.conv1_ch2(x)
        x2 = self.bn1_ch2(x2)
        x2 = self.act_2(x2)

        # 将pool1_ch1和pool1_ch2拼接
        fu_0 = self.fu_0(x1, x2)
        # print('fu_0 shape:', fu_0.shape)
        x1 = self.bottle2neck_ch1_0(x1)
        # x1 = self.CoordAtt1_ch1_0(x1)
        # print('x1 shape:', x1.shape)
        x1 = self.gcm_ch1_0(x1)
        # print('x1-1 shape:', x1.shape)
        # x1 = self.res_ch1_1(x1)

        x2 = self.bottle2neck_ch2_0(x2)
        # x2 = self.CoordAtt1_ch2_0(x2)
        # print('x2 shape:', x2.shape)
        x2 = self.gcm_ch2_0(x2)
        # print('x2-2 shape:', x2.shape)
        # x2 = self.res_ch2_1(x2)

        fu_1 = self.fu_1(x1, x2, fu_0)
        # print('fu_1 shape:', fu_1.shape)
        # pool_fu_1 = self.pool_fu_1(fu_1)
        # print('pool_fu_1 shape:', pool_fu_1     .shape)
        gcm_0 = self.gcm_0(fu_1)
        # print('gcm_0 shape:', gcm_0.shape)
        x1 = self.bottle2neck_ch1_1(x1)
        # x1 = self.CoordAtt1_ch1_1(x1)
        # print('x1-2 shape:', x1.shape)
        # x1 = self.res_ch1_2(x1)
        x1 = self.gcm_ch1_1(x1)
        # print('x1-3 shape:', x1.shape)
        x2 = self.bottle2neck_ch2_1(x2)
        # x2 = self.CoordAtt1_ch2_1(x2)
        # print('x2-2 shape:', x2.shape)
        x2 = self.gcm_ch2_1(x2)
        # print('x2-3 shape:', x2.shape)
        fu_2 = self.fu_1(x1, x2, gcm_0)
        # print('fu_2 shape:', fu_2.shape)
        gcm_1 = self.gcm_1(fu_2)
        # print('gcm_1 shape:', gcm_1.shape)
        x1 = self.bottle2neck_ch1_2(x1)
        # x1 = self.CoordAtt1_ch1_2(x1)
        x1 = self.gcm_ch1_2(x1)
        # print('x1 shape:', x1.shape)
        # x1 = self.mfa_ch1_2(x1)
        x2 = self.bottle2neck_ch2_2(x2)
        # x2 = self.CoordAtt1_ch2_2(x2)
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
        x = torch.cat([x1, x2, gcm_2], dim=1)
        # print('x shape:', x.shape)
        x = self.bn_2(x)
        x = self.act_3(x)
        # x=self.at1(x)
        x = self.pool(x).squeeze()
        # print('x shape:', x.shape)9989

        # 将最后一层的输出通过全连接层得到最终的输出
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
import torch
import torch.nn as nn

class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = self.global_avg_pool(x)
        x_se = x_se.view(x_se.size(0), -1)
        x_se = self.fc(x_se)
        x_se = x_se.view(x_se.size(0), x.size(1), 1)
        return x * x_se

class MCFRNNSET(nn.Module):
    def __init__(self):
        super(MCFRNNSET, self).__init__()
        # 定义第一个通道的卷积层
        self.conv1_ch1 = nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=1)
        self.bn1_ch1 = nn.BatchNorm1d(64)
        self.act_1 = SEAttention(64)
        # self.res_ch1_1=Res2Net(Bottle2neck, [1, 1, 1, 1],)
        self.bottle2neck_ch1_0 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_0=SEAttention(64)
        self.gcm_ch1_0 = GCM(64,64)
        # self.mfam_ch1_0 = MFAM(64, 64)
        self.bottle2neck_ch1_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_1 = SEAttention(64)
        # self.res_ch1_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch1_1 = MFA(4)
        self.gcm_ch1_1 = GCM(64, 64)
        # self.mfam_ch1_1 = MFAM(64, 64)
        self.bottle2neck_ch1_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch1_2 = SEAttention(64)
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
        self.CoordAtt1_ch2_0 = SEAttention(64)
        self.gcm_ch2_0 = GCM(64,64)
        # self.mfam_ch2_0 = MFAM(64, 64)
        self.bottle2neck_ch2_1 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch2_1= SEAttention(64)
        self.gcm_ch2_1 = GCM(64, 64)
        # self.mfam_ch2_1 = MFAM(64, 64)
        # self.res_ch2_2 = Res2Net(Bottle2neck, [1, 1, 1, 1], )
        # self.mfa_ch2_1 = MFA(4)
        self.bottle2neck_ch2_2 = Bottle2neck(inplanes=64, planes=16, stride=1)
        self.CoordAtt1_ch2_2 = SEAttention(64)
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
        self.fc = nn.Linear(192, 5)

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
        # print('x shape:', x.shape)9989


        # 将最后一层的输出通过全连接层得到最终的输出
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
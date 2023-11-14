import torch.nn as nn
from oneD_Meta_ACON import MetaAconC
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvQuadraticOperation import ConvQuadraticOperation
from torch.nn import Linear, BatchNorm1d, ReLU, Sigmoid, AvgPool1d, Conv1d, Softmax

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

class RNNWDCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(RNNWDCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channel),
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 , out_features=100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=out_channel),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
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

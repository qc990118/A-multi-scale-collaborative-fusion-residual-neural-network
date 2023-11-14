import math

import torch
import torch.nn as nn
import numpy as np
from datasave import train_loader, test_loader
from early_stopping import EarlyStopping
from label_smoothing import OLSR,LSR
from oneD_Meta_ACON import MetaAconC
import time
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

    def __init__(self, block, layers ,baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
# #SE模块
# class SEModule(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input):
#         x = self.avg_pool(input)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return input * x
#
#
# class Res2NetBottleneck(nn.Module):
#     expansion = 4  #输出通道数=输入通道数*expansion
#     def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=True,  norm_layer=True):
#         # scales为残差块中使用分层的特征组数，groups表示其中3*3卷积层数量，SE模块和BN层
#         super(Res2NetBottleneck, self).__init__()
#
#         if planes % scales != 0:#输出通道数为4的倍数
#             raise ValueError('Planes must be divisible by scales')
#         if norm_layer:#BN层
#             norm_layer = nn.BatchNorm1d
#
#         bottleneck_planes = groups * planes
#         self.scales = scales
#         self.stride = stride
#         self.downsample = downsample
#         # 1*1的卷积层,在第二个layer时缩小图片尺寸
#         self.conv1 = nn.Conv1d(inplanes, bottleneck_planes, kernel_size=1, stride=stride)
#         self.bn1 = norm_layer(bottleneck_planes)
#         # 3*3的卷积层，一共有3个卷积层和3个BN层
#         self.conv2 = nn.ModuleList([nn.Conv1d(bottleneck_planes // scales, bottleneck_planes // scales,
#                                               kernel_size=3, stride=1, padding=1, groups=groups) for _ in range(scales-1)])
#         self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
#         # 1*1的卷积层，经过这个卷积层之后输出的通道数变成
#         self.conv3 = nn.Conv1d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=1)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         # SE模块
#         self.se = SEModule(planes * self.expansion) if se else None
#
#     def forward(self, x):
#         identity = x
#         # 1*1的卷积层
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         # scales个(3x3)的残差分层架构
#         xs = torch.chunk(out, self.scales, 1)
#         ys = []
#         for s in range(self.scales):
#             if s == 0:
#                 ys.append(xs[s])
#             elif s == 1:
#                 ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
#             else:
#                 ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
#         out = torch.cat(ys, 1)
#         # 1*1的卷积层
#         out = self.conv3(out)
#         out = self.bn3(out)
#         # 加入SE模块
#         if self.se is not None:
#             out = self.se(out)
#         # 下采样
#         if self.downsample:
#             identity = self.downsample(identity)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class Res2Net(nn.Module):
#     def __init__(self, layers, num_classes, width=16, scales=4, groups=1,
#                  zero_init_residual=True, se=True, norm_layer=True):
#         super(Res2Net, self).__init__()
#         if norm_layer:  #BN层
#             norm_layer = nn.BatchNorm1d
#         #通道数分别为64,128,256,512
#         planes = [int(width * scales * 2 ** i) for i in range(4)]
#         self.inplanes = planes[0]
#
#         #7*7的卷积层，3*3的最大池化层
#         self.conv1 = nn.Conv1d(1, planes[0], kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(planes[0])
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         #四个残差块
#         self.layer1 = self._make_layer(Res2NetBottleneck, planes[0], layers[0], stride=1, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
#         self.layer2 = self._make_layer(Res2NetBottleneck, planes[1], layers[1], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
#         self.layer3 = self._make_layer(Res2NetBottleneck, planes[2], layers[2], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
#         self.layer4 = self._make_layer(Res2NetBottleneck, planes[3], layers[3], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
#         #自适应平均池化，全连接层
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(planes[3] * Res2NetBottleneck.expansion, num_classes)
#
#         #初始化
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         #零初始化每个剩余分支中的最后一个BN，以便剩余分支从零开始，并且每个剩余块的行为类似于一个恒等式
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Res2NetBottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, scales=4, groups=1, se=True, norm_layer=True):
#         if norm_layer:
#             norm_layer = nn.BatchNorm1d
#
#         downsample = None  # 下采样，可缩小图片尺寸
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, downsample, stride=stride, scales=scales, groups=groups, se=se,
#                             norm_layer=norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, scales=scales, groups=groups, se=se, norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#
#         return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = Res2Net(Bottle2neck,[1,2,3,4]).to(device)
model = Net().to(device)
# model.load_state_dict(torch.load('./data7/B0503_AdamP_AMS_Nb.pt'))
# for m in model.modules():
#     if isinstance(m, nn.Conv1d):
#         #nn.init.normal_(m.weight)
#         #nn.init.xavier_normal_(m.weight)
#         nn.init.kaiming_normal_(m.weight)
#         #nn.init.constant_(m.bias, 0)
#     # elif isinstance(m, nn.GRU):
#     #     for param in m.parameters():
#     #         if len(param.shape) >= 2:
#     #             nn.init.orthogonal_(param.data)
#     #         else:
#     #             nn.init.normal_(param.data)
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0, std=torch.sqrt(torch.tensor(1/30)))
# input = torch.rand(20, 1, 1024).to(device)
# output = model(input)
# print(output.size())
# with SummaryWriter(log_dir='logs', comment='Net') as w:
#      w.add_graph(model, (input,))
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', 'logs'])
# url = tb.launch()
summary(model, input_size=(1, 1024))

# criterion = nn.CrossEntropyLoss()
# criterion = LSR()
criterion = LSR()
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
from AdamP_amsgrad import AdamP
optimizer = AdamP(model.parameters(), lr=0.001, weight_decay=0.001, nesterov=True, amsgrad=True)
# from adabelief_pytorch import AdaBelief
# optimizer = AdaBelief(model.parameters(), lr=0.001, weight_decay=0.0001, weight_decouple=True)
# from ranger_adabelief import RangerAdaBelief
# optimizer = RangerAdaBelief(model.parameters(), lr=0.001, weight_decay=0.0001, weight_decouple=True)
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
torch.save(model.state_dict(), 'weight/mixcnn.pt')
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
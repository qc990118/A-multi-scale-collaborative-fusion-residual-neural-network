import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

def target_category_loss(x, category_index, nb_classes):
    return torch.mul(x, F.one_hot(category_index, nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)

def resize_1d(array, shape):
    res = np.zeros(shape)
    if array.shape[0] >= shape:
        ratio = array.shape[0]/shape
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        res = res[::-1]
        array = array[::-1]
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        res = res[::-1]/(2*ratio)
        array = array[::-1]
    else:
        ratio = shape/array.shape[0]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = res[::-1]
        array = array[::-1]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = res[::-1]/2
        array = array[::-1]
    return res

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        # print(output.size())
        return output[target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            output = output.squeeze()
            target_category = np.argmax(output.cpu().data.numpy())
            # print(output)
            # print(target_category)
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        #weights = np.mean(grads, axis=(0))
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        # cam = np.zeros(activations.shape[1:], dtype=np.float32)
        # for i, w in enumerate(weights):
        #      cam += w * activations[i, :]
        cam = activations.T.dot(weights)
        # cam = activations.dot(weights)
        # cam = activations.dot(weights)
        # print(input_tensor.shape[1])
        # print(cam.shape)
        # x = np.arange(0, 247, 1)
        # plt.plot(x, cam.reshape(-1, 1))
        # sns.set()
        # ax = sns.heatmap(cam.reshape(-1, 1).T)
        #cam = cv2.resize(cam, input_tensor.shape[1:][::-1])
        #cam = resize_1d(cam, (input_tensor.shape[2]))
        cam = np.interp(np.linspace(0, cam.shape[0], input_tensor.shape[2]), np.linspace(0, cam.shape[0], cam.shape[0]), cam)
        #cam = np.maximum(cam, 0)
        # cam = np.expand_dims(cam, axis=1)
        # ax = sns.heatmap(cam)
        # plt.show()
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)
        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))# + 1e-10   归一化处理
        # heatmap = (cam - np.mean(cam, axis=-1)) / (np.std(cam, axis=-1) + 1e-10)
        # print(heatmap.shape)
        return heatmap
        # return cam
class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(activations, axis=1)
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=1)
        return weights
# from pytorch_grad_cam.utils.image import preprocess_image
model = DoubleChannelNet()
model.load_state_dict(torch.load('weight/MCFRNN.pt'))   ######
target_layer = model.CoordAtt1_ch2_2
net = GradCAM(model, target_layer)
from datasave import Test
from sklearn.preprocessing import MinMaxScaler
# from scipy.fftpack import fft
input_tensor = Test.Data[100:101, :]
ss = MinMaxScaler()
data1 = input_tensor.squeeze().reshape(1, -1).T
data1 = ss.fit_transform(data1).T
data1 = input_tensor.float()
#plt.figure(figsize=(5, 1))
output = net(data1)
# import scipy.io as scio
# import matplotlib
# import matplotlib.pyplot as plt
# color = plt.cm.hsv(output)
# input_tensor = input_tensor.numpy().squeeze()
# time = np.linspace(0, 1023, 1024)
# Line = matplotlib.lines.Line2D(time, input_tensor, color=color)
# plt.plot(*Line.get_data())
# plt.show()
#######################################################
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm

x = np.linspace(0, 1023, 1024)
y = input_tensor.numpy().squeeze()
dydx = output  # first derivative

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap=plt.cm.cool, norm=norm)   #plt.cm.get_cmap('autumn_r')
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)
# line = axs[0].add_collection(lc)
# fig.colorbar(line, ax=axs[0])
line = axs.add_collection(lc)
fig.colorbar(line, ax=axs)

# Use a boundary norm instead
# cmap = ListedColormap(['r', 'g', 'b'])
# norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
# lc = LineCollection(segments, cmap=cmap, norm=norm)
# lc.set_array(dydx)
# lc.set_linewidth(2)
# line = axs[1].add_collection(lc)
# fig.colorbar(line, ax=axs[1])

axs.set_xlim(x.min()-10, x.max()+10)
axs.set_ylim(y.min()-0.05, y.max()+0.05)

plt.savefig("fig1.pdf")
plt.show()
##############################################################
# dataNew = "G:\\datanew.mat"
# scio.savemat(dataNew, mdict={'cam': output, 'data': input_tensor})
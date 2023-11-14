import torch
from torch.utils import data as da
#from hhhh import DoubleChannelNet
from hhhh import DoubleChannelNet
from torchvision.models._utils import IntermediateLayerGetter
activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
def hook(model,input_,output):
    print("最后一层输出：",output.shape)



hyj_model=DoubleChannelNet()

print(hyj_model)
hyj_model.load_state_dict(torch.load('weight/DoubleChannelNet1-OLSR-SGDP-bn-32-5.pt'))
hyj_model.eval()
hyj_model.conv1_ch1.register_forward_hook(get_activation('hhh'))
hyj_model()
k=activation['hhh']
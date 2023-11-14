import torch

weights_files = 'weight/DoubleChannelNet1-OLSR-SGDP-bn-32-5.pt'  # 权重文件路径
weights = torch.load(weights_files)  # 加载权重文件

for k, v in weights.items():  # key, value
    print(k, v)  # 打印参数名、参数值
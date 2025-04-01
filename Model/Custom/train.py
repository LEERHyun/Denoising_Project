from model import HybridNAFNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary 
from ptflops import get_model_complexity_info
from torchvision import transforms
import data
from data import DenoiseDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HybridNAFNet()
model.to(device)
inp_shape = (3,256,256)
macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)
params = float(params[:-3])
macs = float(macs[:-4])
torchsummary.summary(model,inp_shape)

print(f"Custom MACS: {macs}, PARAMS:{params}")

#Init Dataset
dataset_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\Patched_Dataset"  

dataset = DenoiseDataset(root_dir=dataset_dir, transform=None)



#split Data
batch_size = 16
data_size = len(dataset)
train_size = int(0.8*data_size)
test_size = data_size - train_size

from torch.utils.data import random_split
train_dataset, test_dataset = random_split(dataset, [train_size,test_size])

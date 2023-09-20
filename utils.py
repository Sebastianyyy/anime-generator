import torch
image_size = 64
channels = 3
nz = 100
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
batch_size = 128
lr = 0.002
beta1 = 0.5
num_epochs = 60

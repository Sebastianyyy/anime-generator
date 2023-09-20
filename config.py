import torch
import os

class Config():
    def __init__(self, image_size, channels, nz, batch_size, lr, beta1, num_epochs):
        self.image_size = image_size
        self.channels = channels
        self.nz = nz
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        self.num_epochs = num_epochs
        self.current_directory = os.getcwd()

from .data import CustomImageDataset
from .utils import transform
import torch
from config import Config
import torchvision.transforms as transforms
from config import Config

if __name__=='main':
    config = Config(image_size=64, channels=3, nz=100,
                    batch_size=128, lr=0.002, beta1=0.5, num_epochs=40)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data=CustomImageDataset("data",transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=True)

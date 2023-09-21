from model import Generator, Discriminator
from utils import load_weights, show_images, config
import torch

def generate(netD, netG, number_of_images, config):
    netD.eval()
    netG.eval()
    with torch.no_grad():
        noise = torch.randn(number_of_images, config.nz,
                            1, 1).to(config.device)
        fake = netG(noise)
        if config.device == 'cpu':
            fake = torch.Tensor.cpu(fake)
        show_images(fake, number_of_images)
        
number_of_images=16

netG = Generator(config).to(config.device)
print(netG)

netD = Discriminator(config).to(config.device)
print(netD)

load_weights(netD, netG, "weights/model_weightsD.pth",
             "weights/model_weightsG.pth", config)

generate(netD, netG, number_of_images=number_of_images, config=config)



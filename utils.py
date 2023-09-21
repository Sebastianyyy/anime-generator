from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
from config import Config

config = Config(image_size=64, channels=3, nz=100,
                batch_size=128, lr=0.002, beta1=0.5, num_epochs=40)


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(
        make_grid((images.detach()[:nmax])*0.5+0.5, nrow=8).permute(1, 2, 0))
    plt.show()
    

def save_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(
        make_grid((images.detach()[:nmax])*0.5+0.5, nrow=8).permute(1, 2, 0))
    plt.savefig("anime.png")


def show_batch(dl, nmax=64, number_of_batch=0):
    it = iter(dl)
    for _ in range(number_of_batch):
        next(it)
    show_images(next(it), nmax)


def load_weights(model_d, model_g, weights_d, weights_g,config):
    model_d.load_state_dict(torch.load(weights_d,map_location=torch.device(config.device)))
    model_g.load_state_dict(torch.load(weights_g,map_location=torch.device(config.device)))
    
    
def plot_loss(D_losses, G_losses):
    plt.figure(figsize=(25, 15))
    plt.title("Generator Loss and Discriminator Loss")
    plt.plot(D_losses, label="D")
    plt.plot(G_losses, label="G")
    plt.ylabel("Loss")
    plt.xlabel("iterations")
    plt.legend()
    plt.show()



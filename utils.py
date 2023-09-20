from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(
        make_grid((images.detach()[:nmax])*0.5+0.5, nrow=8).permute(1, 2, 0))
    plt.show()


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


def generate(netD,netG,number_of_images, config):
    netD.eval()
    netG.eval()
    with torch.no_grad():
        noise = torch.randn(number_of_images, config.nz, 1, 1).to(config.device)
        fake = netG(noise)
        if config.device=='cpu':
            fake=torch.Tensor.cpu(fake)
        show_images(fake,number_of_images)
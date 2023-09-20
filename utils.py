from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(
        make_grid((images.detach()[:nmax])*0.5+0.5, nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64, number_of_batch=0):
    it = iter(dl)
    for _ in range(number_of_batch):
        next(it)
    show_images(next(it), nmax)

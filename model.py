import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.out = nn.Sequential(
            # channels x image_size x image_size
            nn.Conv2d(config.channels, config.image_size, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.image_size),
            nn.LeakyReLU(0.2),
            # 64 x 32 x 32

            nn.Conv2d(config.image_size, config.image_size*2, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.image_size*2),
            nn.LeakyReLU(0.2),
            # 128 x 16 x 16

            nn.Conv2d(config.image_size*2, config.image_size*4, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.image_size*4),
            nn.LeakyReLU(0.2),
            # 256 x 8 x 8

            nn.Conv2d(config.image_size*4, config.image_size*8, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.image_size*8),
            nn.LeakyReLU(0.2),
            # 512 x 4 x 4

            nn.Conv2d(config.image_size*8, 1, kernel_size=4,
                      stride=1, padding=0, bias=False),
            # 1 x 1 x 1
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.out(x)


class Generator(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.out = nn.Sequential(
            # 100x1x1
            nn.ConvTranspose2d(in_channels=config.nz, out_channels=config.image_size*8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(config.image_size*8),
            nn.ReLU(),
            # 512 x 4 x 4
            nn.ConvTranspose2d(in_channels=config.image_size*8, out_channels=config.image_size * 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.image_size * 4),
            nn.ReLU(),

            # 256 x 8 x 8

            nn.ConvTranspose2d(in_channels=config.image_size*4, out_channels=config.image_size * \
                               2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.image_size * 2),
            nn.ReLU(),
            # 128 x 16 x 16
            nn.ConvTranspose2d(in_channels=config.image_size*2, out_channels=config.image_size,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.image_size),
            nn.ReLU(),
            # 64 x 32 x 32
            nn.ConvTranspose2d(in_channels=config.image_size, out_channels=config.channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            # 3 x 64 x 64
            nn.Tanh()
        )

    def forward(self, x):
        x = self.out(x)
        return x

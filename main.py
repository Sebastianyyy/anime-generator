from data import CustomImageDataset
from utils import load_weights, show_batch, show_images, plot_loss, generate
import torch
from config import Config
import torchvision.transforms as transforms
from config import Config
from model import Generator, Discriminator
 

if __name__ == '__main__':
    config = Config(image_size=64, channels=3, nz=100,
                    batch_size=128, lr=0.002, beta1=0.5, num_epochs=40)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),#changing the range from [0,1] to [-1,1]
    ])

    data=CustomImageDataset("data",transform,config=config)
    data_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=True)

    netG = Generator(config).to(config.device)
    print(netG)

    netD = Discriminator(config).to(config.device)
    print(netD)

    load_weights(netD, netG, "weights/model_weightsD.pth",
                 "weights/model_weightsG.pth",config)

    loss=torch.nn.BCELoss()
    real_label = 1.
    fake_label = 0.
    
    optimizerD = torch.optim.Adam(
        netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = torch.optim.Adam(
        netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    G_losses = []
    D_losses = []
    num_epochs = 1
    netD.train()
    netG.train()
    generate(netD,netG,1,config)
    '''for epoch in range(num_epochs):
        for i, data in enumerate(data_loader, 0):
            # TRAINING DISCRIMINATOR REAL
            netD.zero_grad()
            label = torch.full((config.batch_size,), real_label,
                            dtype=torch.float).to(config.device)
            logits = netD(data.to(config.device)).view(-1)
            error_D_real = loss(logits, label)
            error_D_real.backward()
            E_Dx = logits.mean().item()

            # TRAINING DISCRIMINATOR FAKE
            noise = torch.randn(config.batch_size, config.nz,
                                1, 1).to(config.device)
            fake = netG(noise)
            label.fill_(fake_label)
            logits = netD(fake.detach()).view(-1)
            error_D_fake = loss(logits, label)
            error_D_fake.backward()
            E_DGz_1 = logits.mean().item()
            errorD = error_D_real + error_D_fake
            optimizerD.step()

            # TRAINING GENERATOR
            netG.zero_grad()
            label.fill_(real_label)
            logits = netD(fake).view(-1)
            errorG = loss(logits, label)
            errorG.backward()
            E_DGz_2 = logits.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(f"""[{epoch}/{num_epochs}] [{i}/{len(data_loader)}] Loss Discriminator Real: {error_D_real:.2f}
                Loss Discriminator Fake: {error_D_fake:.2f} 
                Loss Discriminator: {errorD:.2f} 
                Loss Generator: {errorG:.2f} 
                Mean Dx: {E_Dx:.2f}
                Mean Dgz_1: {E_DGz_1:.2f} 
                Mean Dgz_2: {E_DGz_2:.2f}""")

            G_losses.append(errorG.item())
            D_losses.append(errorD.item())

    '''
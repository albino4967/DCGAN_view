import os
from PIL import Image
from torchvision import transforms
import numpy as np
import dataset
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
import torch.nn as nn
import Generator
import Discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



def main() :
    IMAGE_SIZE = 512

    train_data_dir = "dataset"
    data_loader = dataset.generate_dataset(train_data_dir, batch_size=2)

    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
    dataset.plt_show_data(data_loader, device)

    netG = Generator.generator(IMAGE_SIZE, weights_init, device)
    netD = Discriminator.discriminator(IMAGE_SIZE, weights_init, device)

    real_label = 1
    for i, item in enumerate(data_loader) :
        print(item.shape)
        print(f'item : {item}')
        real_cpu = item.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label,
                           dtype=torch.float, device=device)

        print(real_cpu.shape)
        print(f'real_cput : {real_cpu}')
        print(label.shape)
        if i == 0 : break

if __name__ == "__main__":
    main()



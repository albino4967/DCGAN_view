import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

class ViewImageDataset(Dataset) :
    def __init__(self, transform, data_dir):
        self.trainsform = transform
        self.data_dir = data_dir
        self.img_list = list(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file_name = os.path.join(self.data_dir, self.img_list[idx])
        image = Image.open(img_file_name).convert("RGB")
        if self.trainsform is not None :
            img_data = self.trainsform(image)

        return img_data

def generate_dataset(data_path, batch_size):
    data_transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ViewImageDataset(data_transform, data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def plt_show_data(data_loader, device):
    real_batch = next(iter(data_loader))
    print(f'image-data-shape : {real_batch.shape}')
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:1024], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig("train_data_sample.jpg")
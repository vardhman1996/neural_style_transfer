from PIL import Image
import matplotlib.pyplot as plt
import glob
import torchvision.transforms as transforms
import numpy as np
from settings import *
import torch


# Data loader
class ImageFolderLoader(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):

        self.transform = transform
        self.data_paths = []
        for img_path in glob.glob(folder + "/*.jpg"):
            self.data_paths += [img_path]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        img = image_loader(img_path)
        return img

def image_loader(image_name):
    loader = transforms.Compose([
            transforms.Resize(IMSIZE),
            transforms.RandomCrop(IMSIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255.0))
    ])

    image = Image.open(image_name).convert('RGB')
    image = loader(image)
    return image

def imshow(data, title=None, filename='test.jpg'):
    img = data.cpu().clone().clamp(0, 255).numpy()
    img = img.squeeze(0)      # remove the fake batch dimension
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

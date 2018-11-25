from PIL import Image
import matplotlib.pyplot as plt
import glob
import torchvision.transforms as transforms
from settings import *


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
            transforms.ToTensor()])

    image = Image.open(image_name)

    image = loader(image)
    
    return image.to(DEVICE, torch.float)


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

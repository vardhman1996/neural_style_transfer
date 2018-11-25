import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import image_transform_net as img_net
import vgg_net
import data_preprocessing as dp
from settings import *
import multiprocessing

style_img = dp.image_loader("./images/styles/picasso.jpg").unsqueeze(0)
# content_img = dp.image_loader("./images/content/dancing.jpg")

# plt.figure()
# dp.imshow(style_img, title='Style Image')

# plt.figure()
# dp.imshow(content_img, title='Content Image')
# input_img = content_img.clone() #torch.randn(1,3,128,128)
# print("input image shape", input_img.shape)

def train():
    data_train = dp.ImageFolderLoader(IMAGE_FOLDER)

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if USE_CUDA else {}
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    vgg_pretrained = models.vgg16_bn(pretrained=True).features.to(device).eval()

    image_transform_net = img_net.ImageTransformation()
    optimizer = optim.Adam(image_transform_net.parameters())

    for _ in range(EPOCHS):
        for batch_idx, input_img in enumerate(train_loader):
            input_img = input_img.to(DEVICE)
            print(input_img.shape)
            optimizer.zero_grad()
            y_hat = image_transform_net(input_img)

            loss = loss_net(vgg_pretrained, input_img, style_img, y_hat)
            print("Itr: {} Loss: {}".format(batch_idx, loss.item()))
            loss.backward()
            optimizer.step()

    return image_transform_net

def eval(model):
    content_img = dp.image_loader("./images/content/dancing.jpg").unsqueeze(0)

    output = model(content_img)
    output = torch.sigmoid(output)

    plt.figure()
    dp.imshow(output, title='Output Image')

    plt.show()
    
def loss_net(cnn, content_img, style_img, input_img, style_weight=1000000, content_weight=1):
    # print('Building the style transfer model..')
    loss_model, style_losses, content_losses = vgg_net.get_style_model_and_losses(cnn, style_img, content_img)
    
    loss_model(input_img)
    style_score = 0
    content_score = 0

    for sl in style_losses:
        style_score += sl.loss
    for cl in content_losses:
        content_score += cl.loss

    style_score *= style_weight
    content_score *= content_weight

    return style_score + content_score

model = train()
eval(model)
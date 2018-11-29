import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import torchvision.models as models
from collections import namedtuple



from settings import *
import utils

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = utils.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = utils.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        img = img.div_(255.0)
        return (img - self.mean) / self.std


class VGG_loss_net(nn.Module):
    def __init__(self):
        super(VGG_loss_net, self).__init__()
        vgg_pretrained = models.vgg16_bn(pretrained=True).features.to(DEVICE).eval()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained[x])

        vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
        vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
        self.normalize = Normalization(vgg_normalization_mean, vgg_normalization_std)


    def forward(self, X):
        h = self.normalize(X)
        h = self.slice1(h)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out





def get_style_model_and_losses(cnn,
                               style_img, content_img,
                               content_layers=CONTENT_LAYERS_DEFAULT,
                               style_layers=STYLE_LAYERS_DEFAULT):

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(DEVICE)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
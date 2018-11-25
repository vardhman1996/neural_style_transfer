import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ImageTransformation(nn.Module):

    def __init__(self):
        super(ImageTransformation, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        self.bn6 = nn.BatchNorm2d(3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))
        x = F.relu(self.res3(x))
        x = F.relu(self.res4(x))
        x = F.relu(self.res5(x))
        
        x = F.interpolate(x, mode='nearest', scale_factor=2)
#         print("After res5", x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
#         print("After conv4", x.shape)
        x = F.interpolate(x, mode='nearest', scale_factor=2)
        x = F.relu(self.bn5(self.conv5(x)))
#         print("After conv5", x.shape)
        x = self.bn6(self.conv6(x))
#         print("After conv6", x.shape)
#         x = ((torch.tanh(x) + 1) / 2.0)
        return x
        
class ResidualBlock(nn.Module):
    def __init__(self, chan):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(chan)
        self.conv2 = nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(chan)
        
    def forward(self, x):
        residual = x
        
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        
        x = self.bn2(self.conv2(x))
        
        return x + residual
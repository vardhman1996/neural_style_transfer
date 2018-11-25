import torch

USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMSIZE = 512 if torch.cuda.is_available() else 256  # use small size if no gpu


IMAGE_FOLDER = 'images/val2017'

CONTENT_LAYERS_DEFAULT = ['relu_7']
STYLE_LAYERS_DEFAULT = ['relu_2', 'relu_4', 'relu_7', 'relu_10']

# train params
BATCH_SIZE = 1
EPOCHS = 2
LEARNING_RATE = 0.001
MOMENTUM = 0.9
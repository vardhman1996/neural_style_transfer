from comet_ml import Experiment
import time
start = time.time()
import torch
print(time.time() - start)

USE_LOGGER = False

if USE_LOGGER:
    LOGGER = Experiment(api_key="f8rDRoriwkKaL9xSpv7HrpcMT",
                        project_name="fast_neural_style_transfer", workspace="vardhman1996")

USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMSIZE = 420 if torch.cuda.is_available() else 256  # use small size if no gpu

STYLE = 'mosaic'
EVAL_CONTENT_IMAGE = 'amber'

IMAGE_FOLDER = 'images/val2014'
ITERS = 1000

CONTENT_LAYERS_DEFAULT = ['relu_7']
STYLE_LAYERS_DEFAULT = ['relu_2', 'relu_4', 'relu_7', 'relu_10']

# train params
HYPERPARAMETERS = {
    "learning_rate": 0.001,
    "batch_size": 1 if IMSIZE == 420 else 8,
    "epochs": 3,
    "style": STYLE,
    "content": EVAL_CONTENT_IMAGE
}

CHECKPOINT_PATH = 'checkpoints/'
VIDEO_INPUT = 'videos/input'
VIDEO_OUT = 'videos/output'
IMAGE_OUT = 'images/eval'
IMAGE_INPUT = 'images/content'
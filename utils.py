import os
from settings import *
import pickle as pkl
import matplotlib.pyplot as plt

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def make_checkpoint_dir(name):
    dir_path = os.path.join(CHECKPOINT_PATH, name)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path

def write_log(log, filename):
    with open(filename, "wb") as file:
        pkl.dump(log, file)

def read_log(filename):
    with open(filename, "rb") as file:
        return pkl.load(file)

def plot_log(filename):
    log = read_log(filename)
    plt.plot(log['index_log'], log['total_loss_log'], label='total_loss_log')
    plt.plot(log['index_log'], log['style_loss_log'], label='style_loss_log')
    plt.plot(log['index_log'], log['content_loss_log'], label='content_loss_log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.dirname(filename))
    plt.close()

#def gram_matrix(input):
#     a, b, c, d = input.size()  # a=batch size(=1)
#     # b=number of feature maps
#     # (c,d)=dimensions of a f. map (N=c*d)
#
#     features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
#
#     G = torch.mm(features, features.t())  # compute the gram product
#
#     # we 'normalize' the values of the gram matrix
#     # by dividing by the number of element in each feature maps.
#     return G.div(a * b * c * d)
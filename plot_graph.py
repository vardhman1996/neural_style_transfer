import utils
from settings import *
import os

name = 'run2'
style_path = os.path.join(STYLE, name)
checkpoint_path = utils.make_checkpoint_dir(style_path)
log_filename = os.path.join(checkpoint_path, "log.pkl")
utils.plot_log(log_filename)
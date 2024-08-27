import numpy as np
import warnings
import torch
import tensorflow as tf
import random
from utility.builder import Builder

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(0)
tf.random.set_seed(0)
torch.manual_seed(0)
random.seed(0)

DATASET_NAME = "xjtu"

if __name__ == "__main__":
    builder = Builder(DATASET_NAME, bootstrap=0)
    builder.build_new_dataset()
    
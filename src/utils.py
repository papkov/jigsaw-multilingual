import random
import numpy as np
import torch
import os

def seed_everything(seed=54):
    """https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

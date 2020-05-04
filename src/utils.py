import random
import numpy as np
import torch
import os
from tqdm import tqdm 

def seed_everything(seed=54):
    """https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def tqdm_loader(loader, **kwargs):
    """
    Returns input DataLoader wrapped with tqdm(enumerate(loader))
    tqdm params: ascii=True, position=0 (accepting other params as kwargs)
    """
    return tqdm(enumerate(loader), total=len(loader), ascii=True, position=0, **kwargs)


def accuracy(y, pred):
    """
    Calculates accuracy from one-hot encoded labels and predictions 
    """
    return (y.argmax(1) == pred.argmax(1)).mean()
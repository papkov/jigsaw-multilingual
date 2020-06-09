import random
import numpy as np
import torch
import os
from tqdm import tqdm 
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.special import softmax
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns

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

def auc_score(y, pred):
    sm = softmax(pred, 1)
    return roc_auc_score(y[:,1], sm[:,1])


def prediction_analysis(smpred, valid):
    """Prints acc anc auc by lang, plots score distribution and calibration curve"""
    acc = accuracy_score(valid.y, np.round(smpred))
    auc = roc_auc_score(valid.y, smpred)
    print(f'    acc {acc:.4f}, auc {auc:.4f}\n')

    for lang in valid.dataset['lang'].unique():
        mask = valid.dataset['lang'] == lang
        acc = accuracy_score(valid.y[mask], np.round(smpred[mask]))
        auc = roc_auc_score(valid.y[mask], smpred[mask])
        print(f'{lang}: acc {acc:.4f}, auc {auc:.4f}')
        
    fig, ax = plt.subplots(ncols=2, figsize=(7,3))
    
    ax[0].set_title('Score distribution')
    sns.distplot(smpred[valid.y == 0], ax=ax[0], label='non-toxic')
    sns.distplot(smpred[valid.y == 1], ax=ax[0], label='toxic')
    ax[0].legend()
    
    ax[1].set_title('Calibration plot')
    ax[1].plot(*calibration_curve(valid.dataset['toxic'], smpred), 's-')
    plt.show()

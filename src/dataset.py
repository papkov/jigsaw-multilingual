import os
import sys
import torch.utils.data as D
import torch

import pandas as pd
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler


class Dataset(D.Dataset):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.dataset = np.load(fn)
        self.x = self.dataset['x']
        self.y = self.dataset['y']
        self.n_classes = 2
        self.attention_mask = self.dataset['attention_mask']
        
    def process_x(self, x):
        """What to do with x before batching (for augmentations)"""
        return x

    def process_y(self, y):
        """What to do with y before batching (for encoding)"""
        # one-hot encoding
        y = np.eye(self.n_classes, dtype=np.uint8)[y]
        return y
        
    def __getitem__(self, i):
        x, y, am = self.x[i], self.y[i], self.attention_mask[i]
        x = self.process_x(x)
        y = self.process_y(y)
        x, y, am = map(lambda t: torch.tensor(t).long(), [x, y, am])
        return x, y, am
    
    def __len__(self):
        return len(self.y)
    
    def weighted_sampler(self):
        labels, counts = np.unique(self.y, return_counts=True)
        weights = counts[::-1] / counts.sum()
        weights = np.array([weights[i] for i in self.y])
        return WeightedRandomSampler(weights, int(counts.min() * 2))
    
    

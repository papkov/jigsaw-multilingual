import os
import sys
import torch.utils.data as D

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
        self.attention_mask = self.dataset['attention_mask']
        
    def process(self, x):
        return x
        
    def __getitem__(self, i):
        x, am, y = self.x[i], self.attention_mask[i], self.y[i]
        x = self.process(x)
        return x, y, am
    
    def __len__(self):
        return len(self.y)
    
    def weighted_sampler(self):
        labels, counts = np.unique(self.y, return_counts=True)
        weights = counts[::-1] / counts.sum()
        weights = np.array([weights[i] for i in self.y])
        return WeightedRandomSampler(weights, int(counts.min() * 2))
    
    

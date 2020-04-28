import os
import sys
import torch.utils.data as D

import pandas as pd
import numpy as np


class Dataset(D.Dataset):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.dataset = np.load(fn)
        self.x = self.dataset['x']
        self.y = self.dataset['y']
        
    def process(self, x, y):
        return x, y
        
    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        return self.process(x, y)
    
    def __len__(self):
        return len(self.y)
    
    
class BaseAugmentation:
    def __init__(self, p=0.5, bos_token=0, pad_token=1, eos_token=2, dot_token=5):
        self.bos_token = bos_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.dot_token = dot_token
        self.p = p
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
        

class ShuffleSentences(BaseAugmentation):
    """
    Shuffles sentences after splitting by dot token
    """
        
    def __call__(self, sent):
        new_sent = sent.copy()
        
        if np.random.uniform() < self.p:
            # Mask meaningful part of a sentence
            mask = (sent != self.bos_token) & (sent != self.pad_token) & (sent != self.eos_token)
            # Localize dots
            # TODO: a smatrer way to split?
            dots = np.where(sent == self.dot_token)[0]
            # Split by dot localization, shuffle
            split = np.split(sent[mask], dots)
            np.random.shuffle(split)
            # Concatenate back and paste
            new_sent[mask] = np.concatenate(split)
        
        return new_sent
import os
import sys
import torch.utils.data as D

import pandas as pd
import numpy as np
import torch


class BaseTransform:
    def __init__(self, p=0.5, bos_token=0, pad_token=1, eos_token=2, dot_token=5):
        self.bos_token = bos_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.dot_token = dot_token
        self.p = p
        
    def mask(self, sent):
        """Get mask of meaningful tokens"""
        return (sent != self.bos_token) & (sent != self.pad_token) & (sent != self.eos_token)
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
        

class ShuffleSentences(BaseTransform):
    """
    Shuffles sentences after splitting by dot token
    """
        
    def __call__(self, sent):
        new_sent = sent.copy()
        
        if np.random.uniform() < self.p:
            # Mask meaningful part of a sentence
            mask = self.mask(sent)
            # Localize dots
            # TODO: a smatrer way to split?
            dots = np.where(sent == self.dot_token)[0]
            # Split by dot localization, shuffle
            split = np.split(sent[mask], dots)
            np.random.shuffle(split)
            # Concatenate back and paste
            new_sent[mask] = np.concatenate(split)
        
        return new_sent
    

class SwapWords(BaseTransform):
    """
    Swaps words randomly
    """
        
    def __call__(self, sent):
        new_sent = sent.copy()
        
        # Mask meaningful part of a sentence
        mask = self.mask(sent)
        
        # Reverse pair of words in mask
        for i in range(1, len(sent)):
            if np.random.uniform() < self.p and mask[i]:
                new_sent[i-1:i+1] = new_sent[i-1:i+1][::-1]
        
        return new_sent
    
    
class DropWords(BaseTransform):
    """
    Drops words randomly, pastes pad token at the end instead
    """
        
    def __call__(self, sent):
        new_sent = sent.copy()
        
        # Mask meaningful part of a sentence
        mask = self.mask(sent)
        
        # Reverse pair of words in mask
        for i in range(1, len(sent)):
            if np.random.uniform() < self.p and mask[i]:
                new_sent[i:-1] = new_sent[i+1:]
                new_sent[-1] = self.pad_token
        
        return new_sent
    
    
class ToTensor(BaseTransform):
    def __call__(self, x):
        return torch.tensor(x)
    
    
class Compose:
    """
    Composes several transforms together
    """
    def __init__(self, *args):
        self.transforms = args[0]
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
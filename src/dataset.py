import os
import sys
import torch.utils.data as D
import torch

import pandas as pd
import numpy as np


from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from preprocessing import tokenize, clean_text
from transforms import Compose, compose_transforms

from transformers import AutoTokenizer, XLMRobertaTokenizer


def weighted_sampler(y):
    labels, counts = np.unique(y, return_counts=True)
    weights = counts[::-1] / counts.sum()
    weights = np.array([weights[i] for i in y])
    return WeightedRandomSampler(weights, int(counts.min() * 2))


class Dataset(D.Dataset):
    def __init__(self, fn, use_features=False, feature_slice=slice(0,3072), transforms=None):
        super().__init__()
        self.fn = fn
        self.use_features = use_features
        self.feature_slice = feature_slice
        self.dataset = np.load(fn, allow_pickle=True)

        self.features = None
        # TODO customize features?
        if use_features:
            try:
                features_fn = fn.replace('.npz', '')+'_roberta_features.npy'
                self.features = np.load(features_fn)[:,self.feature_slice]
            except:
                print(features_fn, 'loading failed. Set `self.features = None`, use tokenized input')

        self.x = self.features if (use_features and self.features is not None) else self.dataset['x']
        self.y = (self.dataset['y'] > 0.5).astype(np.uint8)
        self.n_classes = 2
        self.attention_mask = self.dataset['attention_mask']
        
        self.transforms = compose_transforms(transforms)
        
    def process_x(self, x):
        """What to do with x before batching (for augmentations)"""
        if self.transforms is not None:
            x = self.transforms(x)
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
        
        # typization
        x, y, am = map(torch.tensor, [x, y, am])
        x = x.float() if self.use_features else x.long()
        y = y.long()
        am = am.long()

        return x, y, am
    
    def __len__(self):
        return len(self.y)
    
    def weighted_sampler(self):
        return weighted_sampler(self.y)
    
    
def make_debug(fn, n=32):
    ds = Dataset(fn)
    to_save = {}
    keys = [k for k in ds.dataset.keys()]
    print(f'Loaded dataset from {fn} with keys {keys}')
    for k in keys:
        to_save[k] = ds.dataset[k][:n]
    np.savez(fn.replace('.npz', '')+f'_debug_{n}.npz', **to_save)


class TokenizerDataset(Dataset):
    def __init__(self, fn, tokenizer_name='xlm-roberta-large', max_length=512, clean=True, text_column='comment_text', transforms=None):
        # Data
        self.dataset = pd.read_csv(fn)
        self.columns = self.dataset.columns
        # Handle different column structure
        assert text_column in self.columns, f'{text_column} is not among columns {self.columns}'
        self.text_column = text_column
        self.y = np.array(self.dataset['toxic'] > 0.5 if 'toxic' in self.columns else np.zeros(len(self.dataset)), dtype=np.uint8)

        self.n_classes = 2
        self.clean = clean

        # Tokenizer
        self.tokenizer_name = tokenizer_name
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transforms = compose_transforms(transforms)


    def __getitem__(self, i):
        x = self.dataset[self.text_column][i]
        y = self.y[i]

        if self.clean:
            x = clean_text(x)

        # Process x and y inherited from base (tokenized) class
        x = self.process_x(x)
        y = self.process_y(y)

        # Get tokenized input and attention mask from tokenizer
        x, am = tokenize([x], self.tokenizer, max_length=self.max_length)
        x, am = x[0], am[0]

        # typization
        x, y, am = map(lambda t: torch.tensor(t).long(), [x, y, am])
        return x, y, am


class ConcatDataset(D.ConcatDataset):
    """Concat dataset implementing weighted sampler"""
    def weighted_sampler(self):
        y = np.concatenate([d.y for d in self.datasets])
        return weighted_sampler(y) 
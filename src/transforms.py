import os
import sys
import torch.utils.data as D

import pandas as pd
import numpy as np
import torch
import random
from copy import deepcopy

from nltk import sent_tokenize

class BaseTransform:
    """
    Base class for text transformations
    Code for text transforms modeified and extended from
    https://www.kaggle.com/shonenkov/nlp-albumentations
    """
    LANGS = {
        'en': 'english',
        'it': 'italian', 
        'fr': 'french', 
        'es': 'spanish',
        'tr': 'turkish', 
        'ru': 'russian',
        'pt': 'portuguese'
    }

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace
        # debug variables storing last actions
        self.last_call = None
        self.was_applied = False

    def get_sentences(self, text, lang='en'):
        # TODO optimize to call once in a series of transforms
        return sent_tokenize(text, self.LANGS.get(lang, 'english'))

    def __call__(self, x):
        try:
            self.last_call = x
            xp = x if self.inplace else deepcopy(x)
            if np.random.uniform() < self.p:
                self.was_applied = True
                return self.apply(xp)
            return xp
        except Exception as e:
            print(f'{type(self)} was failed to apply to [{x}] due to [{e}]')
            return x

    def apply(self, x):
        raise NotImplementedError


class ShuffleSentences(BaseTransform):

    def apply(self, x):
        # TODO pass language
        sentences = self.get_sentences(x)
        random.shuffle(sentences)
        return ' '.join(sentences)


class PickRandomSentence(BaseTransform):

    def apply(self, x):
        # TODO pass language
        sentences = self.get_sentences(x)
        return np.random.choice(sentences)


class PickLastSentence(BaseTransform):

    def apply(self, x):
        # TODO pass language
        sentences = self.get_sentences(x)
        return sentences[-1]

class PickFirstAndLastSentence(BaseTransform):

    def apply(self, x):
        # TODO pass language
        sentences = self.get_sentences(x)
        if len(sentences) > 1:
            sentences = [sentences[0], sentences[-1]] 
        return ' '.join(sentences)


class DropFirstSentence(BaseTransform):

    def apply(self, x):
        # TODO pass language
        sentences = self.get_sentences(x)
        if len(sentences) > 1:
            sentences = sentences[1:]
        return ' '.join(sentences)


class DropRandomSentence(BaseTransform):

    def apply(self, x):
        # TODO pass language
        sentences = self.get_sentences(x)
        if len(sentences) > 1:
            sentences.pop(np.random.choice(len(sentences)))
        return ' '.join(sentences)


class Truncate(BaseTransform):

    def __init__(self, p=1, from_begin=32, from_end=64, **kwargs):
        super().__init__(p=p, **kwargs)
        self.from_begin = from_begin
        self.from_end = from_end

    def apply(self, x):

        words = x.split()
        n_words = len(words)

        words_begin = words[:self.from_begin]
        n_words_begin = len(words_begin)
        # Prevent overlapping
        from_end = min(self.from_end, n_words-n_words_begin)
        words_end = words[-from_end:] if from_end > 0 else []

        words = words_begin + words_end
        return ' '.join(words)


class SwapWords(BaseTransform):

    def __init__(self, p=0.5, swap_p=0.1, swap_distance=1, **kwargs):
        super().__init__(p=p, **kwargs)
        # TODO actually use swap distance
        self.swap_p = swap_p
        self.swap_distance = swap_distance

    def apply(self, x):
        words = x.split()

        # Reverse pair of words in mask
        sd = self.swap_distance
        for i in range(1, len(words)):
            if np.random.uniform() < self.swap_p:
                words[i-1:i+1] = words[i-1:i+1][::-1]
        
        return ' '.join(words)
            

class DropWords(BaseTransform):
    
    def __init__(self, p=0.5, drop_p=0.05, **kwargs):
        super().__init__(p=p, **kwargs)
        self.drop_p = drop_p
    
    def apply(self, x):
        words = x.split()

        i = 0
        while i < len(words):
            if np.random.uniform() < self.drop_p:
                words.pop(i)
                continue
            i += 1

        return ' '.join(words)

class TokenBaseTransform(BaseTransform):
    """
    Base class for token-based transformations
    """
    def __init__(self, p=0.5, bos_token=0, pad_token=1, eos_token=2, dot_token=5, **kwargs):
        super().__init__(p=p, **kwargs)
        self.bos_token = bos_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.dot_token = dot_token
        
    def mask(self, sent):
        """Get mask of meaningful tokens"""
        return (sent != self.bos_token) & (sent != self.pad_token) & (sent != self.eos_token)
        

class TokenShuffleSentences(TokenBaseTransform):
    """
    Shuffles sentences after splitting by dot token
    """
        
    def apply(self, sent):
        # Mask meaningful part of a sentence
        mask = self.mask(sent)
        # Localize dots
        # TODO: a smatrer way to split?
        dots = np.where(sent == self.dot_token)[0]
        # Split by dot localization, shuffle
        split = np.split(sent[mask], dots)
        np.random.shuffle(split)
        # Concatenate back and paste
        sent[mask] = np.concatenate(split)
        
        return sent
    

class TokenSwapWords(TokenBaseTransform):
    """
    Swaps words randomly
    """
    def __init__(self, p=0.5, swap_p=0.1, swap_distance=1, **kwargs):
        super().__init__(p=p, **kwargs)
        # TODO actually use swap distance
        self.swap_p = swap_p
        self.swap_distance = swap_distance
        
    def apply(self, sent):
        # Mask meaningful part of a sentence
        mask = self.mask(sent)
        
        # Reverse pair of words in mask
        for i in range(len(sent)-1):
            if np.random.uniform() < self.swap_p and mask[i]:
                sent[i:i+2] = sent[i:i+2][::-1]
        
        return sent
    
    
class TokenDropWords(TokenBaseTransform):
    """
    Drops words randomly, pastes pad token at the end instead
    """    
    def __init__(self, p=0.5, drop_p=0.1, **kwargs):
        super().__init__(p=p, **kwargs)
        self.drop_p = drop_p
        
    def apply(self, sent):
        # Mask meaningful part of a sentence
        mask = self.mask(sent)
        
        # Reverse pair of words in mask
        for i in range(1, len(sent)):
            if np.random.uniform() < self.drop_p and mask[i]:
                sent[i:-1] = sent[i+1:]
                sent[-1] = self.pad_token
        
        return sent
    
    
class ToTensor(BaseTransform):
    def __call__(self, x):
        return torch.tensor(x)
    
    
class Compose:
    """
    Composes several transforms together
    """
    def __init__(self, *args):
        self.transforms = args
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def compose_transforms(transforms=None):
    """Safely converse list of transforms to Compose object"""
    if transforms is not None:
        if isinstance(transforms, Compose): 
            return transforms
        elif isinstance(transforms, list):
            return Compose(*transforms)
        else:
            raise ValueError
    else:
        return None
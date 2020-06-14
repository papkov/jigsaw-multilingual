from torch import nn
import torch
import numpy as np


class Mix:
    """
    Base class for mix augmentation, property of a model
    Implements linear interpolation and coefficients (lambda) generation
    """
    def __init__(self, alpha=0.4):
        self.idx = None
        self.lam = None
        self.set_alpha(alpha)
    
    def set_alpha(self, alpha):
        "Instantiate beta distribution with `alpha` constant"
        self.beta = torch.distributions.beta.Beta(alpha, alpha)

    def update(self, batch_size):
        """
        Generates new lambda and permutation index for `batch_size`, stores them in `self.idx` and `self.lam`
        """
        # sample indices
        self.idx = torch.randperm(batch_size).long()
        # sample lambda
        lam = self.beta.sample(torch.tensor([batch_size]))
        # ensure that all the lambda elements are > 0.5
        self.lam, _ = torch.stack([lam, 1-lam], 0).max(0)

    def interpolate(self, tensor):
        """
        Performs linear interpolation of tensor (e.g. one-hot encoded labels) with last generated lambda and permutation index
        Can be called externaly from where labels are within the scope, e.g. as `model.mixup.interpolate(y)`
        """
        if self.lam is None:
            raise ValueError
        
        # transfer to the same device where y is stored
        lamd = self.lam.to(tensor.device)
        # unsqueeze lam to match y shape in other dimensions (except batch)
        for i, s in enumerate(tensor.shape[1:]):
            lamd.unsqueeze_(-1)
        # mix x and its permuted version
        return tensor * lamd + tensor[self.idx] * (1-lamd)

    def __call__(self, x):
        "Method to overload"
        raise NotImplementedError


class Mixup(Mix):
    """
    Performs linear interpolation within a 
    """
    def __call__(self, x):
        # generate new lambda and permutation index
        self.update(batch_size=x.shape[0])
        # in mixup mixing for x and y are identical, so we can pass `x` to `interpolate`
        return self.interpolate(x)


class Cutmix(Mix):
    """
    Cut and paste `lambda` ratio from one vector to another 
    """
    def __init__(self, alpha=0.4, continuous=True, inplace=False):
        super().__init__(alpha=alpha)
        self.continuous = continuous
        self.mask = None
        self.inplace = inplace

    def __call__(self, x):
        # generate new lambda and permutation index
        self.update(batch_size=x.shape[0])
        # cut and paste by generated mask
        self.mask = self.generate_mask(x)
        
        x_new = x if self.inplace else x.clone()
        x_new[self.mask] = x[self.idx][self.mask]
        return x_new

    def generate_mask(self, x):
        """2D random mask according to lambda"""
        mask = np.ones_like(x.detach().cpu().numpy(), dtype=bool)
        bs, size = x.shape
        for i, lam in enumerate(self.lam.squeeze()):
            prop = int(lam * size)
            mask[i,:prop] = False
            if self.continuous:
                mask[i] = np.roll(mask[i], np.random.choice(size))
            else:
                np.random.shuffle(mask[i,:])
        return torch.tensor(mask)


class ComposeMix(Mix):
    def __init__(self, alpha=0.4, *args):
        super().__init__(alpha=alpha)
        self.mixes = args[0]
        for m in self.mixes:
            m.update = self.update
            m.lam = self.lam
            m.idx = self.idx
            b.beta = self.beta

    def __call__(self, x):
        x_new = x.clone()
        for m in self.mixes:
            x_new = m(x_new)
        return x



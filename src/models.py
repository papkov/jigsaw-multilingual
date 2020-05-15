from torch import nn
import torch

from mix import *

class SimplePoolingHead(nn.Module):
    def __init__(self, in_features=3072, out_features=2, dropout=0.5, mix=None):
        super().__init__()
        # mixup parameters
        self.in_features = in_features
        self.out_features = out_features
        self.mix = mix

        self.head = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args):
        # TODO identify where mixup is most useful
        if self.mix is not None and self.training:
            x = self.mix(x)

        x = self.dropout(x)
        x = self.head(x)
        return x


class CustomPoolingHead(SimplePoolingHead):
    """AUC 0.906, do not touch"""
    def __init__(self, dropout=0.5, *args, **kwargs):
        super().__init__(dropout=dropout, *args, **kwargs)
        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.in_features),
            torch.nn.Linear(self.in_features, self.in_features, bias=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm1d(self.in_features),
            torch.nn.Linear(self.in_features, self.out_features, bias=False)
        )


class TransformersPoolingHead(SimplePoolingHead):
    def __init__(self, dropout=0.5, relu=False, *args, **kwargs):
        super().__init__(dropout=dropout, *args, **kwargs)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.in_features),
            torch.nn.ReLU() if relu else torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.in_features, self.out_features)
        )


class Model(nn.Module):

    def __init__(self, backbone, mix=None, head=SimplePoolingHead()):
        super().__init__()
        # mixup parameters
        self.mix = mix

        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, attention_masks):
        batch_size, max_len = input_ids.shape
        x, _ = self.backbone(input_ids=input_ids, attention_mask=attention_masks)
        
        # use mixup only if provided and when training
        # TODO: before of inside the head?
        if self.mix is not None and self.training:
            x = self.mix(x)

        # two poolings for feature extraction
        pool_avg = torch.mean(x, 1)
        pool_max, _ = torch.max(x, 1)
        cls_token = x[:,0,:]
        x = torch.cat((cls_token, pool_avg, pool_max), 1)
        
        x = self.head(x)
        return x



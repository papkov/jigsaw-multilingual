from torch import nn
import torch
from tqdm import tqdm

from mix import *

# from transformers.modeling_utils import get_extended_attention_mask

class SimplePoolingHead(nn.Module):
    def __init__(self, in_features=3072, out_features=2, dropout=0.5, mix=None, freeze_bn=False, activation=None, bias=True):
        super().__init__()
        # mixup parameters
        self.in_features = in_features
        self.out_features = out_features
        self.mix = mix
        self.freeze_bn = freeze_bn
        self.activation = activation

        self.bias = bias
        self.head = nn.Linear(in_features=in_features, out_features=out_features, bias=self.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args):
        # Input activation
        if self.activation == 'tanh':
            x = torch.tanh(x)
        elif self.activation == 'relu':
            x = torch.nn.functional.relu(x)

        # TODO identify where mixup is most useful
        if self.mix is not None and self.training:
            x = self.mix(x)

        x = self.dropout(x)
        x = self.head(x)
        return x

    def _freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()

    def train(self, mode=True):
        super().train(mode=mode)
        if self.freeze_bn:
            tqdm.write('Freeze BatchNorm in the head (set to eval)')
            self._freeze_bn()


class BNHead(SimplePoolingHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.in_features),
            torch.nn.Linear(self.in_features, self.out_features, bias=self.bias)
            )


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

    def __init__(self, backbone, mix=None, mix_pos='sent', head=SimplePoolingHead(), features=['cls', 'avg', 'max']):
        super().__init__()
        # mixup parameters
        assert mix_pos in ['sent', 'word']
        self.mix_pos = mix_pos
        self.mix = mix

        self.backbone = backbone
        self.head = head
        self.features = features

    def forward(self, input_ids, attention_mask):
        batch_size, max_len = input_ids.shape

        inputs_embeds = self.backbone.embeddings(input_ids=input_ids)
        if self.mix is not None and self.training and self.mix_pos == 'word':
            # word mixup only if provided and when training
            inputs_embeds = self.mix(inputs_embeds)

        x, _ = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # x, _ = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # two poolings for feature extraction
        pool_avg = torch.mean(x, 1)
        pool_max, _ = torch.max(x, 1)
        cls_token = x[:,0,:]
        features = dict(cls=cls_token, avg=pool_avg, max=pool_max)
        x = torch.cat([features[f] for f in self.features], 1)

        
        if self.mix is not None and self.training and self.mix_pos == 'sent':
            # sentence mixup only if provided and when training
            x = self.mix(x)
        
        x = self.head(x)
        return x



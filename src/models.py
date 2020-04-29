from torch import nn
import torch

class Model(nn.Module):

    def __init__(self, backbone, dropout=0.25, mix=False, alpha=0.4):
        super().__init__()
        # mixup parameters
        self.mix = mix
        self.mixup = Mixup(alpha)
        
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            in_features=self.backbone.pooler.dense.out_features*2,
            out_features=2,
        )

    def forward(self, input_ids, attention_masks):
        batch_size, max_len = input_ids.shape
        x, _ = self.backbone(input_ids=input_ids, attention_mask=attention_masks)
        
        idx = None
        lam = None
        # use mixup only if provided and when training
        # TODO: before of after pooling?
        if self.mix and self.training:
            x, idx, lam = self.mixup(x)
        
        pool_avg = torch.mean(x, 1)
        pool_max, _ = torch.max(x, 1)
        
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)
        x = self.linear(x)
        
        return x, idx, lam
    

class Mixup:
    def __init__(self, alpha=0.4):
        self.set_alpha(alpha)
    
    def set_alpha(self, alpha):
        self.beta = torch.distributions.beta.Beta(alpha, alpha)
        
    def __call__(self, x):
        # Permute x in batch anong 0th axis
        bs = x.shape[0]
        idx = torch.randperm(bs)
        x_permuted = x[idx]
        # sample lambda, ensure that all the elements are > 0.5
        lam = self.beta.sample(torch.tensor([bs]))
        lam, _ = torch.stack([lam, 1-lam], 0).max(0)
        lam = lam[..., None]
        # mix x and its permuted version
        x_permuted = x * lam + x_permuted * (1-lam)
        
        # return everything for mixing labels as well
        return x_permuted, idx, lam
        
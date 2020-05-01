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
        
        # use mixup only if provided and when training
        # TODO: before of after pooling?
        if self.mix and self.training:
            x = self.mixup(x)
        
        pool_avg = torch.mean(x, 1)
        pool_max, _ = torch.max(x, 1)
        
        x = torch.cat((pool_avg, pool_max), 1)
        x = self.dropout(x)
        x = self.linear(x)
        
        return x
    

class Mixup:
    def __init__(self, alpha=0.4):
        self.idx = None
        self.lam = None
        self.set_alpha(alpha)
    
    def set_alpha(self, alpha):
        self.beta = torch.distributions.beta.Beta(alpha, alpha)
        
    def __call__(self, x):
        bs = x.shape[0]
        idx = torch.randperm(bs)

        # sample lambda
        lam = self.beta.sample(torch.tensor([bs]))
        # ensure that all the elements are > 0.5
        lam, _ = torch.stack([lam, 1-lam], 0).max(0)
        # unsqueeze lam to match x shape in other dimensions (except batch) 
        for i, s in enumerate(x.shape[1:]):
            lam.unsqueeze_(-1)
        
        # store last idx and lam for mixing y
        self.idx = idx
        self.lam = lam.squeeze()

        # mix x and its permuted version
        lamd = lam.to(x.device)
        return x * lamd + x[idx] * (1-lamd)

    def mix_y(self, y):
        if self.lam is None:
            raise ValueError
        
        # unsqueeze lam to match y shape in other dimensions (except batch)
        for i, s in enumerate(y.shape[1:]):
            self.lam.unsqueeze_(-1)

        lamd = self.lam.to(y.device)
        return y * lamd + y[self.idx] * (1-lamd)
        
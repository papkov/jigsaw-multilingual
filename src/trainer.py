import torch
from torch import nn
from tqdm import tqdm 
from torch.utils import data as D
import torch.nn.functional as F
import numpy as np

import transformers
from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

from copy import deepcopy
from utility import accuracy, auc_score, tqdm_loader

try:
    from apex import amp
    AMP_AVAILABLE = True
except ModuleNotFoundError:
    print('Install apex for mixed precision training: https://github.com/NVIDIA/apex')
    AMP_AVAILABLE = False



class Meter:
    """Stores all the incremented elements, their sum and average"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.history = []
        self.sum = 0
        self.avg = 0
        self.last = 0
        self.min = np.inf
        self.max = -np.inf
        self.extremum = ''

    def add(self, value):
        """Increment overriden to store new value"""
        self.last = value
        self.extremum = ''

        if value < self.min: 
            self.min = value
            self.extremum = 'min'
        if value > self.max:
            self.max = value
            self.extremum = 'max'

        self.history.append(value)
        self.sum += value
        self.avg = self.sum / len(self.history)

    def is_best(self):
        """Check if the last epoch was the best according to the meter"""
        is_best = (self.name.endswith('loss') and self.extremum == 'min') or \
                  ((self.name.endswith('acc') or self.name.endswith('auc')) and self.extremum == 'max')

        return is_best

def ce_loss(x, target):
    x = x.float()
    target = target.float()
    logprobs = torch.nn.functional.log_softmax(x, dim=-1)

    loss = -logprobs * target
    return loss.sum(-1)

class DenseCrossEntropy(nn.Module):
    """Cross-entropy for one-hot encoded targets"""
    def __init__(self, label_smoothing=0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, x, target):
        if self.label_smoothing > 0:
            # supposes 2 classes
            target = (1-self.label_smoothing) * target + self.label_smoothing / 2
        loss = ce_loss(x, target)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        loss = ce_loss(x, target)
        pt = torch.exp(-loss)
        loss = self.alpha * (1 - pt)**self.gamma * loss
        return loss.mean()


class Trainer(nn.Module):
    def __init__(self, name, model, loader_train, loader_valid, loader_test=None, device='cuda', epochs=5, 
                 gradient_accumulation=1, opt_level=None,
                 monitor='val_auc', checkpoint_path='../checkpoints/',
                 **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.device = device
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path

        assert opt_level in [None, 'O0', 'O1', 'O2', 'O3']
        self.opt_level = opt_level
        self.amp = self.opt_level is not None and AMP_AVAILABLE


        # Monitoring
        self.monitor = monitor

        # Data loaders
        self.loader_train = loader_train
        self.loader_valid = loader_valid
        self.loader_test = loader_test

        # Gradient accumutation
        # TODO sync normalization
        self.gradient_accumulation = gradient_accumulation

        # Optimization
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5) if 'optimizer' not in kwargs else kwargs['optimizer']
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5) if 'optimizer' not in kwargs else kwargs['optimizer']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0, last_epoch=-1) if 'scheduler' not in kwargs else kwargs['scheduler']
        self.criterion = DenseCrossEntropy() if 'criterion' not in kwargs else kwargs['criterion']

        self.meters = {m:Meter(m) for m in ['loss', 'val_loss', 'val_acc', 'val_auc']}

        # Mixed precision
        if self.amp:
            print(f'Use automatic mixed precision at opt_level={self.opt_level}')
            self.model.to(self.device)
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opt_level)


    def forward(self, x, y, attention_masks, *args):
        """Handles transfer to device, computes loss"""
        output = self.model(x.to(self.device), attention_masks.to(self.device), *args)
        if self.model.mix is not None and self.model.training:
            # mix y if model performed mix and is in train mode
            y = self.model.mix.interpolate(y)
        loss = self.criterion(output, y.to(self.device))
        return output, loss

        
    def one_epoch(self, epoch_id=0):
        self.model.train()
        self.optimizer.zero_grad()

        loss_meter = Meter('loss')
        acc_meter = Meter('acc')
        progress_dict = dict(loss=0, acc=0)
        lr = self.scheduler.get_last_lr()[-1]
        iterator = tqdm_loader(self.loader_train, desc=f'ep. {epoch_id:04d} (lr {lr:.02e})', postfix=progress_dict)
        for i, batch in iterator:
            output, loss = self.forward(*batch)

            if self.amp:
                # Handle auto mixed precision
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()   

            # Make step once in `self.gradient_accumulation` batches 
            if (i + 1) % self.gradient_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # calculate batch accuracy
            output = output.detach().cpu().numpy()
            y = batch[1].cpu().numpy()
            acc = accuracy(y, output)
            
            # logging
            loss = loss.item()
            loss_meter.add(loss)
            acc_meter.add(acc)
            iterator.set_postfix(dict(loss=loss_meter.avg, acc=acc_meter.avg))

        # Post-epoch actions
        self.scheduler.step()

        return loss
        
    def fit(self, epochs=None):
        # TODO distributed training and TPU
        self.model = self.model.to(self.device)

        if epochs is None:
            epochs = self.epochs

        try:
            # Training
            for epoch in range(epochs):
                # Train one epoch
                loss = self.one_epoch(epoch)
                self.meters['loss'].add(loss)

                # Validate
                output, val_loss, val_acc, val_auc = self.validate()
                self.meters['val_loss'].add(val_loss)
                self.meters['val_acc'].add(val_acc)
                self.meters['val_auc'].add(val_auc)

                # Show val results
                status = ', '.join([f'{name}={meter.last:.4f}' for name, meter in self.meters.items()])
                tqdm.write(f'Epoch {epoch} complete. {status}')

                # Save checkpoint
                name = self.name
                if not self.meters[self.monitor].is_best():
                    name += '_last'
                self.save_checkpoint(name=name, epoch=epoch)
            
            # Test
            # if self.loader_test is not None:
            #     return self.test()

        except KeyboardInterrupt:
            tqdm.write('Interrupted')

    def predict(self, loader, desc='predict'):
        self.model.eval()
        preds = []
        ys = []
        loss_meter = Meter('loss')
        with torch.no_grad():
            for i, batch in tqdm_loader(loader, desc=desc):
                # predict
                output, loss = self.forward(*batch)
                output = output.cpu()

                # log
                loss_meter.add(loss.item())
                
                # store
                # deepcopy handles RuntimeError: received 0 items of ancdata https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
                preds.append(output.numpy())
                ys.append(deepcopy(batch[1].numpy()))

        # calculate prediction accuracy
        preds, ys = map(np.concatenate, [preds, ys])
        acc = accuracy(ys, preds)
        auc = auc_score(ys, preds)
        return preds, loss_meter.avg, acc, auc
    
    def validate(self):
        self.model = self.model.to(self.device)
        return self.predict(self.loader_valid, desc='valid')
    
    def test(self):
        self.model = self.model.to(self.device)
        return self.predict(self.loader_test, desc='test')

    def save_checkpoint(self, epoch=None, name=None):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        # Handle auto mixed precision
        if self.amp:
            state.update({'amp': amp.state_dict(), 'opt_level': self.opt_level})
        # Add metrics to the dict
        state.update({name: meter.last for name, meter in self.meters.items()})
        path = f'{self.checkpoint_path}/{self.name if name is None else name}.pth'
        torch.save(state, path)
        tqdm.write(f'Saved model to {path}')
        
    def load_checkpoint(self, path=None):
        path =  f'{self.checkpoint_path}/{self.name}.pth' if path is None else path
        state = torch.load(path)
        status = ''
        for key, value in state.items():
            if key in ['model', 'optimizer', 'scheduler']:
                getattr(self, key).load_state_dict(value)
            elif key == 'amp':
                if AMP_AVAILABLE:
                    amp.load_state_dict(state['amp'])
                    print(f'Load state in automatic mixed precision mode at opt_level={self.opt_level}')
                    assert self.opt_level == state['opt_level'], f"opt_level of the Trainer should be the same {state['opt_level']}, because amp will not be reinitialized"
                else:
                    print('Checkpoint was saved in automatic mixed precision mode, but it is not available now')
            elif key in ['epoch', 'opt_level']:
                pass
            else:
                self.meters[key].add(value)
                status += f'{key}: {value:.4f} '

        tqdm.write(f'Loaded model from {path}\nepoch {state["epoch"]}, {status}')
        # TODO TPU
        self.model = self.model.to(self.device)
    
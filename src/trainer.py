import torch
from torch import nn
from tqdm import tqdm 
from torch.utils import data as D
import numpy as np

import transformers
from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

def tqdm_loader(loader, **kwargs):
    """
    Returns input DataLoader wrapped with tqdm(enumerate(loader))
    tqdm params: ascii=True, position=0 (accepting other params as kwargs)
    """
    return tqdm(enumerate(loader), total=len(loader), ascii=True, position=0, **kwargs)

def accuracy(y, pred):
    return (y.argmax(1) == pred.argmax(1)).mean()

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
        self.min = 0
        self.max = 0
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
                  (self.name.endswith('acc') and self.extremum == 'max')

        return is_best


class DenseCrossEntropy(nn.Module):
    """Cross-entropy for one-hot encoded targets"""
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class Trainer(nn.Module):
    def __init__(self, name, model, loader_train, loader_valid, loader_test=None, epochs=5, monitor='val_loss', checkpoint_path='../checkpoints/', **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path

        # Monitoring
        self.monitor = monitor

        # Data loaders
        self.loader_train = loader_train
        self.loader_valid = loader_valid
        self.loader_test = loader_test

        # Optimization
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5) if 'optimizer' not in kwargs else kwargs['optimizer']
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5) if 'optimizer' not in kwargs else kwargs['optimizer']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0, last_epoch=-1) if 'scheduler' not in kwargs else kwargs['scheduler'](self.optimizer)
        self.criterion = DenseCrossEntropy() if 'criterion' not in kwargs else kwargs['criterion']

        self.meters = {m:Meter(m) for m in ['loss', 'val_loss', 'val_acc']}


    def forward(self, x, y, attention_masks, *args):
        """Handles transfer to device, computes loss"""
        output = self.model(x.cuda(), attention_masks.cuda(), *args)
        if self.model.mix:
            # mix y if model performed mixup
            y = self.model.mixup.mix_y(y)
        loss = self.criterion(output, y.cuda())
        acc = accuracy(y.numpy(), output.detach().cpu().numpy())
        return output, loss, acc

        
    def one_epoch(self, epoch_id=0):
        self.model.train()

        loss_meter = Meter('loss')
        acc_meter = Meter('acc')
        progress_dict = dict(loss=0, acc=0)
        lr = self.scheduler.get_lr()[-1]
        iterator = tqdm_loader(self.loader_train, desc=f'ep. {epoch_id:04d} (lr {lr:.02e})', postfix=progress_dict)
        for i, batch in iterator:
            # TODO implement gradient accumulation
            self.optimizer.zero_grad()
            output, loss, acc = self.forward(*batch)
            loss.backward()   
            self.optimizer.step()
            
            # logging
            loss = loss.item()
            loss_meter.add(loss)
            acc_meter.add(acc)
            iterator.set_postfix(dict(loss=loss_meter.avg, acc=acc_meter.avg))

        return loss
        
    def fit(self, epochs=None):
        # TODO distributed training and TPU
        self.model = self.model.cuda()

        if epochs is None:
            epochs = self.epochs

        try:
            # Training
            for epoch in range(epochs):
                # Train one epoch
                loss = self.one_epoch(epoch)
                self.meters['loss'].add(loss)

                # Validate
                output, val_loss, val_acc = self.validate()
                self.meters['val_loss'].add(val_loss)
                self.meters['val_acc'].add(val_acc)

                # Show val results
                tqdm.write(f'Epoch {epoch} complete. val loss (avg): {val_loss:.4f}, val acc: {val_acc:.4f}')

                # Save checkpoint
                name = self.name
                if not self.meters[self.monitor].is_best():
                    name += '_last'
                self.save_checkpoint(name=name, epoch=epoch)

                # Post-epoch actions
                self.scheduler.step()
            
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
        acc_meter = Meter('acc')
        with torch.no_grad():
            for i, batch in tqdm_loader(loader, desc=desc):
                # predict
                output, loss, acc = self.forward(*batch)
                output = output.cpu()

                # log
                loss_meter.add(loss.item())
                acc_meter.add(acc)
                
                # store
                preds.append(output.numpy())
                ys.append(batch[1].numpy())

        preds, ys = map(np.concatenate, [preds, ys])
        acc = accuracy(ys, preds)
        return preds, loss_meter.avg, acc
    
    def validate(self):
        return self.predict(self.loader_valid, desc='valid')
    
    def test(self):
        return self.predict(self.loader_test, desc='test')

    def save_checkpoint(self, epoch=None, name=None):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
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
            else:
                if key != 'epoch':
                    self.meters[key].add(value)
                    status += f'{key}: {value:.4f} '

        tqdm.write(f'Loaded model from {path}\nepoch {state["epoch"]}, {status}')
        # TODO TPU
        self.model = self.model.cuda()
    
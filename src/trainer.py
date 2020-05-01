import torch
from torch import nn
from tqdm import tqdm 
from torch.utils import data as D

def tqdm_loader(loader, **kwargs):
    """
    Returns input DataLoader wrapped with tqdm(enumerate(loader))
    tqdm params: ascii=True, position=0 (accepting other params as kwargs)
    """
    return tqdm(enumerate(loader), total=len(loader), ascii=True, position=0, **kwargs)


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


class Trainer(nn.Module):
    def __init__(self, name, model, loader_train, loader_valid, loader_test=None, epochs=5, monitor='val_loss', **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.epoch = epochs

        # Monitoring
        self.monitor = monitor

        # Data loaders
        self.loader_train = loader_train
        self.loader_valid = loader_valid
        self.loader_test = loader_test

        # Optimization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5) if 'optimizer' not in kwargs else kwargs['optimizer']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0, last_epoch=-1) if 'scheduler' not in kwargs else kwargs['scheduler'](self.optimizer)
        self.criterion = torch.nn.CrossEntropyLoss() if 'criterion' not in kwargs else kwargs['criterion']

        self.meters = {m:Meter(m) for m in ['loss', 'val_loss', 'val_acc']}


    def forward(self, x, y, *args):
        """Compute loss"""
        output = self.model(x.cuda())
        loss = self.criterion(output, y.cuda())
        return output, loss

        
    def one_epoch(self, epoch_id=0):
        self.model.train()

        loss_meter = Meter()
        progress_dict = dict(loss=0)
        for i, batch in tqdm_loader(self.loader_train, desc=f'ep. {epoch:04d} (lr {lr:.02e})', postfix=progress_dict):
            # TODO implement gradient accumulation
            self.optimizer.zero_grad()
            output, loss = self.forward(*batch)

            loss.backward()            
            self.optimizer.step()
        
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

                # Save checkpoint
                name = self.name
                if not self.meters[self.monitor].is_best():
                    name += '_last'
                self.save_checkpoint(name)
            
            # Test
            if self.loader_test is not None:
                return self.test()

        except KeyboardInterrupt:
            tqdm.write('Interrupted')

    
    def predict(self, loader, desc='predict'):
        self.model.eval()
        predictions = []
        ys = []

        preds = []
        ys = []
        cum_loss = 0
        with torch.no_grad():
            for i, batch in tqdm_loader(loader, desc=desc):
                output, loss = self.forward(*batch)
                output = output.cpu()
                cum_loss += loss.item()
                preds.append(output.numpy())
                ys.append(batch[1].numpy())

        preds, ys = map(np.concatenate, [preds, ys])
        return preds, cum_loss/len(loader)
    
    def validate(self):
        return self.predict(self.loader_valid, desc='valid')
    
    def test(self):
        return self.predict(self.loader_test, desc='test')

    
    def save_checkpoint(self, name=None):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        # Add metrics to the dict
        state.update({m.name: m.last for m in self.meters})
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
    
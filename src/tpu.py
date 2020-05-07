# not ready

def run():
    # Samplers
    sampler_train = torch.utils.data.distributed.DistributedSampler(
      train,
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=True)
    
    sampler_valid = torch.utils.data.distributed.DistributedSampler(
      valid,
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=False)
    
    # Loaders
    loader_train = torch.utils.data.DataLoader(
        train,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=sampler_train,
        drop_last=True,
        num_workers=4
    )

    loader_valid = torch.utils.data.DataLoader(
        valid,
        batch_size=64,
        sampler=sampler_valid,
        drop_last=False,
        num_workers=4
    )
    
    # Parallelize
    para_loader_train = pl.ParallelLoader(loader_train, [device]).per_device_loader(device)
    para_loader_valid = pl.ParallelLoader(loader_valid, [device]).per_device_loader(device)
    
    # Optimization
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    lr = 0.4 * 1e-5 * xm.xrt_world_size()
    num_train_steps = int(len(train) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    
    # Trainer
    trnr = trainer.Trainer('baseline', 
                           model, para_loader_train, para_loader_valid,
                           device=device, 
                           epochs=EPOCHS, 
                           checkpoint_path='.',
                           optimizer=optimizer,
                           scheduler=scheduler,
                          )
    
    # Fit
    trnr.fit()
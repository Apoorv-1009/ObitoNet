import os
import torch
import torch.optim as optim
from .TanksAndTemples import TanksAndTemples
from utils.misc import *
from models import ObitoNet
from experiments.ply_image_CA import ObitoNetCA
from utils.logging import *
from timm.scheduler import CosineLRScheduler

def dataset_builder(args, config):
    dataset = TanksAndTemples(config)
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

# <TODO> Remove when we split obitonet into ca and pc and img (built seperately)
def model_builder(config):
    model = ObitoNet.ObitoNet(config)
    return model

def obitonet_pc_builder(config):
    model = ObitoNet.ObitoNetPC(config)
    return model

def obitonet_img_builder(config):
    model = ObitoNet.ObitoNetImg(config)
    return model

def obitonet_ca_builder(config):
    model = ObitoNet.ObitoNetCA(config)
    return model

def experiment_model_builder(config):
    model = ObitoNetCA.ObitoNet(config)
    return model

def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                t_mul=1,
                lr_min=1e-6,
                decay_rate=0.1,
                warmup_lr_init=1e-6,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return optimizer, scheduler

def resume_model(base_model, model, args, logger = None):
    if model == 'ObitoNetPC':
        ckpt_path = os.path.join(args.experiment_path, 'obitonet_pc_ckpt-last.pth')
    elif model == 'ObitoNetImg':
        ckpt_path = os.path.join(args.experiment_path, 'obitonet_img_ckpt-last.pth')
    elif model == 'ObitoNetCA':
        ckpt_path = os.path.join(args.experiment_path, 'obitonet_ca_ckpt-last.pth')
    else:
        raise NotImplementedError
    
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict = True)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1}', logger = logger)
    return start_epoch

def resume_optimizer(optimizer, model, args, logger = None):
    if model == 'ObitoNetPC':
        ckpt_path = os.path.join(args.experiment_path, 'obitonet_pc_ckpt-last.pth')
    elif model == 'ObitoNetImg':
        ckpt_path = os.path.join(args.experiment_path, 'obitonet_img_ckpt-last.pth')
    elif model == 'ObitoNetCA':
        ckpt_path = os.path.join(args.experiment_path, 'obitonet_ca_ckpt-last.pth')
    else:
        raise NotImplementedError
    
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, model, args, logger = None):
    if model == 'ObitoNetPC':
        ckpt_path = os.path.join(args.experiment_path, 'obitonet_pc_ckpt-last.pth')
    elif model == 'ObitoNetImg':
        ckpt_path = os.path.join(args.experiment_path, 'obitonet_img_ckpt-last.pth')
    elif model == 'ObitoNetCA':
        ckpt_path = os.path.join(args.experiment_path, 'obitonet_ca_ckpt-last.pth')
    else:
        raise NotImplementedError
    
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict = True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return 

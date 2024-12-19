import torch
import torch.nn as nn
from torchvision import transforms
from models.ObitoNet import *
from utils import data_transforms, dist_utils, misc
from utils.logging import print_log
from tools import builder
import time
import wandb
import os

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class AverageMeter(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]

def train(args, config, device, train_writer=None, val_writer=None):
    logger = None
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)

    # Build Point Cloud Encoder
    obitonet_pc = builder.obitonet_pc_builder(config.model)

    # Build Image Encoder
    obitonet_img = builder.obitonet_img_builder(config.model)

    # Build Cross Attention Decoder
    obitonet_ca = builder.obitonet_ca_builder(config.model)

    # Build ObitoNet
    obitonet = ObitoNet(config.model, obitonet_pc, obitonet_img, obitonet_ca)

    if args.use_gpu:
        obitonet = nn.DataParallel(obitonet)
        obitonet.to(device)
    
    wandb.watch(obitonet_pc, log="all")
    wandb.watch(obitonet_img, log="all")
    wandb.watch(obitonet_ca, log="all")
    wandb.watch(obitonet, log="all")

    # parameter setting
    start_epoch = 0

    # resume ckpts
    if args.resume:
        start_epoch = builder.resume_model(obitonet_pc, 'ObitoNetPC', args, logger = logger)
        start_epoch = builder.resume_model(obitonet_img, 'ObitoNetImg', args, logger = logger)
        start_epoch = builder.resume_model(obitonet_ca, 'ObitoNetCA', args, logger = logger)
    elif args.start_ckpt_epoch is not None:
        start_epoch = int(args.start_ckpt_epoch)
        builder.load_model(obitonet_pc, 'ObitoNetPC', args, logger = logger)
        builder.load_model(obitonet_img, 'ObitoNetImg', args, logger = logger)
        builder.load_model(obitonet_ca, 'ObitoNetCA', args, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            obitonet_pc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(obitonet_pc)
            obitonet_img = torch.nn.SyncBatchNorm.convert_sync_batchnorm(obitonet_img)
            obitonet_ca = torch.nn.SyncBatchNorm.convert_sync_batchnorm(obitonet_ca)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        obitonet_pc = nn.parallel.DistributedDataParallel(obitonet_pc, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        obitonet_img = nn.parallel.DistributedDataParallel(obitonet_img, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        obitonet_ca = nn.parallel.DistributedDataParallel(obitonet_ca, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        obitonet_pc = nn.DataParallel(obitonet_pc).cuda()
        obitonet_img = nn.DataParallel(obitonet_img).cuda()
        obitonet_ca = nn.DataParallel(obitonet_ca).cuda()

    # optimizer & scheduler
    pc_optimizer, pc_scheduler = builder.build_opti_sche(obitonet_pc, config)
    img_optimizer, img_scheduler = builder.build_opti_sche(obitonet_img, config)
    ca_optimizer, ca_scheduler = builder.build_opti_sche(obitonet_ca, config)
    
    if args.resume:
        builder.resume_optimizer(pc_optimizer, 'ObitoNetPC',args, logger = logger)
        builder.resume_optimizer(img_optimizer,'ObitoNetImg', args, logger = logger)
        builder.resume_optimizer(ca_optimizer,'ObitoNetCA', args, logger = logger)

    # Set the gradient to zero
    obitonet_pc.zero_grad()
    obitonet_img.zero_grad()
    obitonet_ca.zero_grad()


    # Freeze all layers except the last (optional, for fine-tuning)
    # for param in obitonet_ca.module.parameters():
    #     param.requires_grad = True

    # Add a new reconstruction head
    # obitonet_ca.module.increase_dim = nn.Sequential(
    #     nn.Conv1d(obitonet_ca.module.trans_dim, 512, 1),  # Hidden layer
    #     nn.ReLU(inplace=True),  # Nonlinearity
    #     nn.Conv1d(512, 3 * obitonet_ca.module.group_size, 1)  # Output layer
    # ).to(device)

    # Unfreeze the new layer
    # for param in obitonet_ca.module.increase_dim.parameters():
    #     param.requires_grad = True



    # Set image encoder to training mode, PC encoder CA to eval modeeval
    obitonet_img.train()
    obitonet_pc.train()
    obitonet_ca.train()

    # obitonet_img.eval()
    # obitonet_pc.eval()
    # obitonet_ca.train()

    # # Freeze the obitonet_pc and obitonet_ca
    # for param in obitonet_pc.parameters():
    #     param.requires_grad = False

    # for param in obitonet_img.parameters():
    #     param.requires_grad = False

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        # Get number of batches
        n_batches = len(train_dataloader)

        for idx, (points, img) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'TanksAndTemples':
                points = points.cuda()
                img = img.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            points = train_transforms(points)
            loss = obitonet(points, img)
            try:
                loss.backward()
                # print("Using one GPU")
            except Exception as e:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0

                # Update the weights
                pc_optimizer.step()
                img_optimizer.step()
                ca_optimizer.step()

                # Set the gradient to zero
                obitonet_pc.zero_grad()
                obitonet_img.zero_grad()
                obitonet_ca.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()*1000])
            else:
                losses.update([loss.item()*1000])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Point Cloud Encoder Optimizer Loss/Batch/LR', pc_optimizer.param_groups[0]['lr'], n_itr)
                train_writer.add_scalar('Image Encoder Optimizer Loss/Batch/LR', img_optimizer.param_groups[0]['lr'], n_itr)
                train_writer.add_scalar('Cross Attention Decoder Optimizer Loss/Batch/LR', ca_optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s pc_lr = %.6f ca_lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], pc_optimizer.param_groups[0]['lr'], ca_optimizer.param_groups[0]['lr']), logger = logger)
        
        if isinstance(pc_scheduler, list):
            for item in pc_scheduler:
                item.step(epoch)
        else:
            pc_scheduler.step(epoch)

        if isinstance(img_scheduler, list):
            for item in img_scheduler:
                item.step(epoch)
        else:
            img_scheduler.step(epoch)

        if isinstance(ca_scheduler, list):
            for item in ca_scheduler:
                item.step(epoch)
        else:
            ca_scheduler.step(epoch)


        epoch_end_time = time.time()
        
        # log metrics to wandb
        wandb.log({"loss": losses.avg(0), 
                    "losses":['%.4f' % l for l in losses.avg()],
                    "pc_encoder_lr":pc_optimizer.param_groups[0]['lr'], 
                    "img_encoder_lr":img_optimizer.param_groups[0]['lr'],
                    "ca_decoder_lr":ca_optimizer.param_groups[0]['lr']})

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s pc_lr = %.6f ca_lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             pc_optimizer.param_groups[0]['lr'], ca_optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % 10 == 0:
            builder.save_checkpoint(obitonet_pc, pc_optimizer, epoch, 'mod_obitonet_pc_ckpt-last', args, logger = logger)
            builder.save_checkpoint(obitonet_img, img_optimizer, epoch, 'mod_obitonet_img_ckpt-last', args, logger = logger)
            builder.save_checkpoint(obitonet_ca, ca_optimizer, epoch, 'mod_obitonet_ca_ckpt-last', args, logger = logger)
            wandb.save(os.path.join(args.experiment_path, 'mod_obitonet_pc_ckpt-last.pth'))
            wandb.save(os.path.join(args.experiment_path, 'mod_obitonet_img_ckpt-last.pth'))
            wandb.save(os.path.join(args.experiment_path, 'mod_obitonet_ca_ckpt-last.pth'))

        # if epoch % 200 ==0 and epoch >=250:
        #     builder.save_checkpoint(obitonet_pc, pc_optimizer, epoch, 'obitonet_pc_ckpt-epoch-{epoch:03d}', args, logger = logger)
        #     builder.save_checkpoint(obitonet_img, img_optimizer, epoch, 'obitonet_img_ckpt-epoch-{epoch:03d}', args, logger = logger)
        #     builder.save_checkpoint(obitonet_pc, ca_optimizer, epoch, 'obitonet_ca_ckpt-epoch-{epoch:03d}', args, logger = logger)
        #     wandb.save(f'obitonet_pc_ckpt-epoch-{epoch:03d}')
        #     wandb.save(f'obitonet_img_ckpt-epoch-{epoch:03d}')
        #     wandb.save(f'obitonet_ca_ckpt-epoch-{epoch:03d}')
 
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
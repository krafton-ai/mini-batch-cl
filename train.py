
#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import sogclr.builder
import sogclr.loader
import sogclr.optimizer
import sogclr.folder

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

import utils
import datetime
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
wandb = None # for debugging

def get_parser(description='Mini-Batch Contrastive Loss Pre-Training'):
    torchvision_model_names = sorted(name for name in torchvision_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision_models.__dict__[name]))

    model_names = torchvision_model_names

    parser = argparse.ArgumentParser(description=description, conflict_handler='resolve')
    parser.add_argument('--data', metavar='DIR', default='/data/cifar100/',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4096, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4096), this is the total '
                            'batch size of all GPUs on all nodes when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-fb', '--feature-batch-size', default=1024, type=int,
                        help='batch size per gpu for features extraction')
    parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                        metavar='W', help='weight decay (default: 1e-6)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save-freq', default=10, type=int,
                        help='save frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument("--wandb_notes", default="", type=str, help="additional wandb logging note")


    # moco specific configs:
    parser.add_argument('--dim', default=128, type=int,
                        help='feature dimension (default: 256)')
    parser.add_argument('--mlp-dim', default=2048, type=int,
                        help='hidden dimension in MLPs (default: 4096)')
    parser.add_argument('--t', default=0.1, type=float,
                        help='softmax temperature (default: 1.0)')
    parser.add_argument('--num_proj_layers', default=2, type=int,
                        help='number of non-linear projection heads')

    # other upgrades
    parser.add_argument('--optimizer', default='lars', type=str,
                        choices=['lars', 'adamw', 'sgd'],
                        help='optimizer used (default: lars)')
    parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('--crop-min', default=0.08, type=float,
                        help='minimum scale for random cropping (default: 0.08)')
    parser.add_argument('--accum_steps', default=1, type=int,
                        help='number of steps to accumulate gradients')
    parser.add_argument('--accum_average', action='store_true')

    # dataset 
    parser.add_argument('--data_name', default='cifar100', type=str) 
    parser.add_argument('--save_dir', default='./logs/', type=str) 


    # simclr
    parser.add_argument('--objective_type', default='sim', 
                        help='Which objective type we use (sim [=simclr], sog [=sogclr], hcl).',
                        choices=['sim', 'sog', 'hcl']  # Do not use "-" or "_"
                        )
    # sogclr
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='for updating moving average estimator u for sogclr')
    parser.add_argument('--learning-rate-scaling', default='sqrt', type=str,
                        choices=['sqrt', 'linear'],
                        help='learing rate scaling (default: sqrt)')

    # batch sampling
    parser.add_argument('--batch_sampling', default='s', type=str,
                        choices=["s", "osgd_kb_loose", "sc_even_kb_loose"],
                        help='batch sampling scheme')
    parser.add_argument('--best_criteria', default='max', type=str,
                        choices=["min", "max"],
                        help='criteria to decide best batch which is sampled by batch sampling algorithm')
    parser.add_argument('--search_subset_ratio', default=None, type=float,
                        help='subset ratio of train dataset for greedy search')
    parser.add_argument('--k', default=None, type=int,
                        help='number of target batches for searching (corresponding to subset of NcB) in osgd_kb')
    parser.add_argument('--q', default=1, type=int,
                        help='number of batches (being used for training from K) in osgd families')
    parser.add_argument('--max-dataset-size', default=None, type=int,
                        help='constrain dataset size for debugging')
    parser.add_argument('--sc-rand-freq', default=2, type=int,
                        help='(TBD))')

    # hcl
    parser.add_argument('--hcl_tau_plus', default=None, type=float,
                        help='HCL\' tau plus (class-prior)')
    parser.add_argument('--hcl_beta', default=None, type=float,
                        help='HCL\' beta')

    parser.add_argument('--bimodal', action='store_true')

    return parser


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    args = get_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        print(f"multiprocessing_distributed, ngpus_per_node={ngpus_per_node}")
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # assertions
    if "kb" in args.batch_sampling:
        assert args.k is not None and args.k >= args.q
        if args.batch_sampling in ["sc_even_kb_loose"]: assert args.k == args.q
        if args.max_dataset_size:
            assert args.max_dataset_size % (args.world_size * args.feature_batch_size) == 0 and \
                args.max_dataset_size >= (args.k * args.batch_size) # args.batch_size is global batch size in distributed setting
    else:
        args.k is None
    if args.objective_type == 'hcl':
        assert (args.hcl_tau_plus != None) and (args.hcl_beta != None)
    else:
        assert (args.hcl_tau_plus == None) and (args.hcl_beta == None)

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, timeout=datetime.timedelta(seconds=3600 * 6), rank=args.rank)
        torch.distributed.barrier()

    # log_dir 
    save_root_path = args.save_dir
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    objective_tag = f"obj_{args.objective_type}"
    if args.objective_type == 'hcl':
        objective_tag += f"(tp{args.t}ta{args.hcl_tau_plus}bt{args.hcl_beta})"
    batch_sampling_tag = args.batch_sampling
    if args.batch_sampling in ["osgd_kb_loose", "sc_even_kb_loose"]:
        batch_sampling_tag = f"{args.batch_sampling}_{args.best_criteria}"
        if "kb" not in args.batch_sampling:
            batch_sampling_tag += f"_ssr{args.search_subset_ratio}"
        else:
            batch_sampling_tag += f"_k{args.k}_q{args.q}"
    group_tag = '230815_%s_%s_bz_%s_accum%s_E%s_lr_%.7f_%s_%s_%s_%s'\
        %(args.data_name, args.arch, args.batch_size, args.accum_steps, args.epochs, args.lr, args.learning_rate_scaling, args.optimizer, objective_tag, batch_sampling_tag)
    if args.max_dataset_size:
        group_tag += f"_mds{args.max_dataset_size}"
    # group_tag = 'TEST'
    exp_name = f"{time_tag}_{group_tag}"
    exp_notes = args.wandb_notes
    if args.rank == 0:
        if wandb:
            print("init wandb logging...")
            
            wandb.init(
                entity="mini_batch_CL",
                project="pretraining",
                group=group_tag,
                name=exp_name,
                notes=exp_notes,
                config=vars(args)
            )
    summary_writer = SummaryWriter(log_dir=os.path.join(save_root_path, group_tag))
    os.makedirs(os.path.join(save_root_path, group_tag), exist_ok=True)
    print("group_tag : ", group_tag)

    # sizes for each dataset
    if args.data_name == 'tiny_imagenet':
        data_size = 100000+1
    elif args.data_name == "cifar100":
        data_size = 50000+1
    elif args.data_name == "mscoco":
        data_size = 118287+1
    # elif args.data_name == "cc3m":
    else:
        data_size = 10000
    print ('pretraining on %s'%args.data_name)

    # create model
    set_all_seeds(2022)
    print("=> creating model '{}'".format(args.arch))
    BIMODAL = args.bimodal
    print(f"Bimodal : {BIMODAL}")
    if not BIMODAL:
        model = sogclr.builder.SimCLR_ResNet(
                partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
                args.dim, args.mlp_dim, args.t, loss_type='dcl', N=data_size, num_proj_layers=args.num_proj_layers)
    else:
        from open_clip import create_model_and_transforms, trace_model
        from attrdict import AttrDict
        # [Reference] https://git.projectbro.com/deep-learning/project-clap/open_clip/-/blob/dev_clif_wandb_exp/src/scripts/clif/221124/run_exp1_80000_2-modal_unfrozen_lr1e8.sh
        from open_clip_config import open_clip_dict
        def get_default_params(model_name):
            # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
            model_name = model_name.lower()
            if "vit" in model_name:
                return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
            else:
                return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
        default_params = get_default_params(open_clip_dict['model'])
        open_clip_dict.update(default_params)
        open_clip_args = AttrDict(open_clip_dict)

        model = sogclr.builder.SimCLR_CLIP(args, open_clip_args, N=data_size)

    # infer learning rate before changing batch size
    if args.learning_rate_scaling == 'linear':
        # infer learning rate before changing batch size
        args.lr = args.lr * args.batch_size / 256
    else:
        # sqrt scaling  
        args.lr = args.lr * math.sqrt(args.batch_size)
        
    print ('initial learning rate:', args.lr)      
    print('objective type: {}'.format(args.objective_type))
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.global_batch_size = args.batch_size * args.world_size
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            if args.bimodal:
                open_clip_args.batch_size = args.batch_size
                open_clip_args.global_batch_size = args.global_batch_size
                open_clip_args.workers = args.workers
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        # pass
    #print(model) # print model after SyncBatchNorm

    # create optimizer and scaler
    if not BIMODAL:
        if args.optimizer == 'lars':
            optimizer = sogclr.optimizer.LARS(model.parameters(), args.lr,
                                            weight_decay=args.weight_decay,
                                            momentum=args.momentum)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)

        scaler = torch.cuda.amp.GradScaler()
    else:
        # https://git.projectbro.com/deep-learning/project-clap/open_clip/-/blame/dev_clif_wandb_exp/src/training/main.py#L177 ~ #L204
        optimizer = None
        scaler = None
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            lr=args.lr,
            betas=(open_clip_args.beta1, open_clip_args.beta2),
            eps=open_clip_args.eps,
        )
        if open_clip_args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if open_clip_args.precision == "amp" else None

    # optionally resume from a checkpoint
    if not BIMODAL:
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                if args.gpu is None:
                    checkpoint = torch.load(args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scaler.load_state_dict(checkpoint['scaler'])
                model.module.u = checkpoint['u'].cpu()
                print('check sum u:', model.module.u.sum())
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # https://git.projectbro.com/deep-learning/project-clap/open_clip/-/blob/dev_clif_wandb_exp/src/training/main.py#L206 ~ #L228
        # optionally resume from a checkpoint
        import logging
        start_epoch = 0
        if args.resume is not None:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                if 'epoch' in checkpoint:
                    # resuming a train checkpoint w/ epoch and optimizer state
                    start_epoch = checkpoint["epoch"]
                    sd = checkpoint["state_dict"]
                    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                        sd = {k[len('module.'):]: v for k, v in sd.items()}
                    model.load_state_dict(sd)
                    if optimizer is not None:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    if scaler is not None and 'scaler' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler'])
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
                else:
                    # loading a bare (model only) checkpoint for fine-tune or evaluation
                    model.load_state_dict(checkpoint)
                    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # TODO: Data Loadiing for bimodal-setting.
    if not BIMODAL:
        # Data loading code
        # for cifar-, refer to https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2851662#gistcomment-2851662
        mean = {'tiny_imagenet': [0.485, 0.456, 0.406],
                'cifar100': [0.5071, 0.4865, 0.4409],
                }[args.data_name]
        std = {'tiny_imagenet': [0.229, 0.224, 0.225],
                'cifar100': [0.2673, 0.2564, 0.2762],
                }[args.data_name]

        image_size = {'tiny_imagenet':224, 'cifar100':224}[args.data_name]
        normalize = transforms.Normalize(mean=mean, std=std)

        # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
        # simclr
        augmentation1 = [
            transforms.RandomResizedCrop(image_size, scale=(args.crop_min, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([sogclr.loader.GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        if args.data_name in ['tiny_imagenet', 'cifar100'] :
            traindir = os.path.join(args.data, 'train')
            train_dataset = sogclr.folder.ImageFolder(
                traindir,
                sogclr.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                                transforms.Compose(augmentation1)),
                                                max_dataset_size=args.max_dataset_size)
        else:
            raise ValueError
    else:
        # TODO: Data Loadiing for bimodal-setting.
        # TODO: Implement MS-COCO / CC3M dataset loading
        from oc_data import get_data
        if hasattr(model, 'module'):
            data = get_data(open_clip_args, (model.module.preprocess_train, model.module.preprocess_val), epoch=start_epoch)
        else:
            data = get_data(open_clip_args, (model.preprocess_train, model.preprocess_val), epoch=start_epoch)
        train_dataset = data['train'].dataset
        print(f"len(train_dataset) : {len(train_dataset)}")

    print('batch_sampling: {}'.format(batch_sampling_tag))
    assert args.feature_batch_size % (args.batch_size) == 0, "Due to drop_last=True."

    # define preemptive loader for fixed and shuffled batch
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False if args.batch_sampling=='f' else True)
    else:
        sampler = None
    shuffle = False if args.batch_sampling=='f' else sampler is None
    preemptive_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=True)

    # define data loader for feature extraction
    feature_loader = None
    if args.batch_sampling not in ['f', 's']:
        if args.distributed:
            feature_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        else:
            feature_sampler = None
        feature_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.feature_batch_size, shuffle=feature_sampler is None,
            num_workers=args.workers, pin_memory=True, sampler=feature_sampler, drop_last=True)


    def _get_average_meters(iters_per_epoch, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        learning_rates = AverageMeter('LR', ':.4e')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, learning_rates, losses],
            prefix="Epoch: [{}]".format(epoch))
        return batch_time, data_time, learning_rates, losses, progress

    iters_per_epoch = len(preemptive_loader) // args.accum_steps
    epoch = args.start_epoch
    step = iters_per_epoch * epoch
    avg_meters = _get_average_meters(iters_per_epoch, epoch)
    # print(f"iters_per_epoch {iters_per_epoch}, len(preemptive_loader): {len(preemptive_loader)}")
    while epoch <= args.epochs:

        # train for one epoch
        start_time = time.time()
        step = train(iters_per_epoch, preemptive_loader, feature_loader, model, optimizer, scaler, summary_writer, epoch, step, avg_meters, args)
        print('elapsed time (s): %.1f'%(time.time() - start_time))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
            
            if epoch % args.save_freq == 0 or args.epochs - epoch < 3:
                local_u = model.module.u
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'u': model.module.u, 
                }, is_best=False, filename=os.path.join(save_root_path, group_tag, 'checkpoint_%04d.pth.tar' % epoch) )
        if step % iters_per_epoch == 0:
            epoch += 1
            avg_meters = _get_average_meters(iters_per_epoch, epoch)
            if args.distributed:
                sampler.set_epoch(epoch)

    if args.rank == 0:
        summary_writer.close()
        if wandb:
            wandb.finish()


def train(iters_per_epoch, preemptive_loader, feature_loader, model, optimizer, scaler, summary_writer, epoch, step, avg_meters, args):
    batch_time, data_time, learning_rates, losses, progress = avg_meters

    # switch to train mode
    model.train()

    end = time.time()

    # sample loader
    with torch.no_grad():
        train_loader, batch_sample_time = utils.sample_loader(preemptive_loader, feature_loader, model, epoch, step, args)
    if batch_sample_time and args.rank == 0:
        if wandb:
            wandb.log({"batch_sample_time": batch_sample_time})

    lr = adjust_learning_rate(optimizer, step / iters_per_epoch, args)
    learning_rates.update(lr)
    optimizer.zero_grad()

    for i, (images, _, index) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], index, args.gamma, args)
            if args.accum_average:
                loss /= args.accum_steps

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        if ((i + 1) % args.accum_steps == 0):

            if args.rank == 0:
                train_logs = {
                    'train/epoch': epoch,
                    'train/step': step,
                    'train/loss': loss.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/data_time': data_time.avg,
                }
                for name, val in train_logs.items():
                    summary_writer.add_scalar(name, val, step)
                if wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log(train_logs)

            # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1

            if step % args.print_freq == 0:
                # print(f"step : {step}, iters_per_epoch * epoch: {iters_per_epoch * epoch}, step - iters_per_epoch * epoch : {step - iters_per_epoch * epoch}")
                progress.display(step - iters_per_epoch * epoch)
                print(f"update! at i={i + 1}")

            if step % iters_per_epoch == 0:
                break

            # adjust learning rate and momentum coefficient per iteration
            lr = adjust_learning_rate(optimizer, step / iters_per_epoch, args)
            learning_rates.update(lr)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return step


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()

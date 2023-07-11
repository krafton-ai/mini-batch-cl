import os
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from functools import partial
from collections import defaultdict
import datetime
import argparse
import numpy as np
import pickle

import utils
import sogclr.builder
import sogclr.loader
import sogclr.optimizer
import sogclr.folder
from train import set_all_seeds, get_parser
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
# from lincls import AverageMeter
from typing import Any, Callable, List, Optional, Tuple


class CorruptedDatasetFolder(sogclr.folder.DatasetFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (path, target, index) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        return path, target, index

    def collate_fn(self, batch: List[Tuple[str, int, int, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(batch) == 1, f"len(batch)=={len(batch)}, which should be 1. Please check the --eval_data_name."
        images_0, images_1, targets, indices = [], [], [], []

        for path, target, index in batch:
            for tgt_curroption in self.corruptions:
                new_path = path.replace(self.input_curroption, tgt_curroption)

                # Load the image and apply the transform
                sample = self.loader(new_path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                images_0.append(sample[0])
                images_1.append(sample[1])
                targets.append(target)
                indices.append(index)

        images_0 = torch.stack(images_0)
        images_1 = torch.stack(images_1)
        targets = torch.tensor(targets, dtype=torch.long)
        indices = torch.tensor(indices, dtype=torch.long)

        return (images_0, images_1), targets, indices


class CorruptedImageFolder(CorruptedDatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = sogclr.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        max_dataset_size = None,
    ):
        super().__init__(
            root,
            loader,
            sogclr.folder.IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            max_dataset_size=max_dataset_size
        )
        self.imgs = self.samples

        data_root = "/" + "/".join([*root.split("/")[:-2]])
        self.input_curroption = root.split("/")[-2]
        self.corruptions = [crpt for crpt in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, crpt))]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_metrics(args, image_features, text_features, logit_scale, tqdm_desc=False):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image1_to_image2": logits_per_image, "image2_to_image1": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    iterator = logits.items() if not tqdm_desc else tqdm(logits.items(), desc=f'rank[{args.rank}] | get_metrics')
    for name, logit in iterator:
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mrr"] = _get_mrr(ranking)
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def _get_mrr(indices):
    mrr = []
    for ii, inds in enumerate(indices):
        mrr.append(1 / (inds.tolist().index(ii)+1))
    return np.mean(mrr)


def get_args(ckpt_path, epoch, global_args):
    class args:
        resume = ckpt_path
        # dist_backend = "nccl"
        # multiprocessing_distributed = True
        # dist_url = "tcp://224.66.41.62:23456"
        world_size = 1
        gpu = 0
        rank = 0
        dim = 128
        mlp_dim = 2048
        t = 0.1
        loss_type = "dcl"
        num_proj_layers = 2
        gamma = 0.9
        search_subset_ratio = global_args.search_subset_ratio

        # dataset
        workers = 8
        crop_min = 0.08
    args.epoch = epoch

    args.arch = args.resume.split("/")[-2].split("_bz")[0].split("_")[-1]
    args.pretrained_data_name = args.resume.split("/")[-2].split(f"_{args.arch}")[0]
    args.data_name = args.resume.split("/")[-2].split(f"_{args.arch}")[0] \
        if global_args.eval_data_name is None else global_args.eval_data_name
    args.lr = args.resume.split('/')[-2].split("lr_")[1].split("_")[0]
    if args.data_name == "tiny_imagenet":
        args.feature_batch_size = 2048
        args.data = "path/to/data"
    elif args.data_name == "cifar100":
        args.feature_batch_size = 2500
        args.data = "path/to/data"
    elif args.data_name == "tiny_imagenet_c":
        args.feature_batch_size = 2048
        args.data = "path/to/data"
    elif args.data_name == "cifar100_c":
        args.feature_batch_size = 2000
        args.data = "path/to/data"
    if global_args.evaluation == "corrupted_top1_accuracy":
        args.feature_batch_size = 1
    args.batch_size = int(args.resume.split("/")[-2].split("bz_")[1].split("_")[0]) \
        if global_args.batch_size_plotting is None else global_args.batch_size_plotting
    args.global_batch_size = args.batch_size
    args.k, args.q = -1, -1
    if 'accum' in args.resume:
        pretrained_accum_steps = int(args.resume.split("/")[-2].split("accum")[1].split("_")[0])
        args.accum_tag = f"_accum{pretrained_accum_steps}"
    args.optimizer = args.resume.split('/')[-2].split("_obj")[0].split("_")[-1]
    args.objective_type = args.resume.split('/')[-2].split(f"{args.optimizer}_obj_")[1].split("_")[0]
    if "osgd_kb" in args.resume:
        try:
            args.k = int(args.resume.split("/")[-2].split(f"{args.optimizer}_obj_{args.objective_type}_")[1].split("_k")[2].split("_q")[0])
            args.q = int(args.resume.split("/")[-2].split(f"{args.optimizer}_obj_{args.objective_type}_")[1].split("_q")[1])
            args.q = args.q.split("_mds")[0] if "mds" in args.resume else args.q
        except:
            pass
        args.pretrained_batch_sampling = args.resume.split("/")[-2].split(f"{args.optimizer}_obj_{args.objective_type}_")[1].split(f"_k{args.k}")[0]
    else:
        args.pretrained_batch_sampling = args.resume.split("/")[-2].split(f"{args.optimizer}_obj_{args.objective_type}_")[1].split("_mds")[0]
        args.pretrained_batch_sampling = args.pretrained_batch_sampling.split("_mds")[0] if "mds" in args.resume else args.pretrained_batch_sampling
    args.pretrained_batch_sampling += f"_lr{args.lr}"
    max_dataset_size = args.resume.split("/")[-2].split("mds")[1] if "mds" in args.resume else 'None'
    if max_dataset_size == 'None':
        if 'cifar' in args.data_name:
            if global_args.data_pathname == "train":
                args.max_dataset_size = 50000
            else:
                args.max_dataset_size = 10000
        elif 'tiny_imagenet' in args.data_name:
            if global_args.data_pathname == "train":
                args.max_dataset_size = 98304
            else:
                args.max_dataset_size = 8192
        else:
            raise ValueError()
    else:
        args.max_dataset_size = int(max_dataset_size)
    args.distributed = False #args.world_size > 1 or args.multiprocessing_distributed

    # sizes for each dataset
    if args.data_name == 'tiny_imagenet':
        args.data_size = 129395+1
    elif args.data_name == "cifar100":
        args.data_size = 50000+1
    elif args.data_name == 'tiny_imagenet_c':
        args.data_size = 129395+1
    elif args.data_name == "cifar100_c":
        args.data_size = 50000+1
    else:
        args.data_size = 1000000

    return args


def create_model(epoch, args, global_args):
    print("=> creating model '{}'".format(args.arch))
    
    if global_args.evaluation == "linear_evaluation":
        print("=> creating linear model '{}'".format(args.arch))
        if args.arch.startswith('vit'):
            model = sogclr.vits.__dict__[args.arch]()
            linear_keyword = 'head'
        else:
            model = torchvision_models.__dict__[args.arch]()
            linear_keyword = 'fc'
        # remove original fc and add fc with customized num_classes
        hidden_dim = model.fc.weight.shape[1]
        del model.fc  # remove original fc layer
        model.fc = nn.Linear(hidden_dim, args.num_classes, bias=True)
        # print (model)
    else:
        model = sogclr.builder.SimCLR_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
            args.dim, args.mlp_dim, args.t, loss_type=args.loss_type, N=args.data_size, num_proj_layers=args.num_proj_layers)      
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    if epoch >= 0:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            # model.u = checkpoint['u'].cpu()
            # print('check sum u:', model.u.sum())
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            args.epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    return model


def get_train_dataset(args, global_args):
    # for cifar-, refer to https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2851662#gistcomment-2851662
    mean = {'tiny_imagenet': [0.485, 0.456, 0.406],
            'cifar100': [0.5071, 0.4865, 0.4409],
            'tiny_imagenet_c': [0.485, 0.456, 0.406],
            'cifar100_c': [0.5071, 0.4865, 0.4409],
            }[args.pretrained_data_name]
    std = {'tiny_imagenet': [0.229, 0.224, 0.225],
            'cifar100': [0.2673, 0.2564, 0.2762],
            'tiny_imagenet_c': [0.229, 0.224, 0.225],
            'cifar100_c': [0.2673, 0.2564, 0.2762],
        }[args.pretrained_data_name]

    image_size = {'tiny_imagenet':224, 'cifar100':224, 'tiny_imagenet_c':224, 'cifar100_c':224}[args.pretrained_data_name]
    normalize = transforms.Normalize(mean=mean, std=std)

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
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

    if args.data_name in ['tiny_imagenet', 'cifar100', "tiny_imagenet_c", "cifar100_c"] :
        traindir = os.path.join(args.data, global_args.data_pathname)
        if global_args.evaluation == "corrupted_top1_accuracy":
            train_dataset = CorruptedImageFolder(
                traindir,
                sogclr.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                                transforms.Compose(augmentation1)),
                                                max_dataset_size=args.max_dataset_size)
        else:
            train_dataset = sogclr.folder.ImageFolder(
                traindir,
                sogclr.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                                transforms.Compose(augmentation1)),
                                                max_dataset_size=args.max_dataset_size)
    else:
        raise ValueError
    
    return train_dataset


def get_top1_accuracy(epoch, args, model, feature_loader, full_batch=True):
    model.eval()

    if full_batch:
        # Get all features
        images_u_all, images_v_all, indices = [], [], []
        indices, images_u_all, images_v_all = utils.get_learned_features(feature_loader, model, 1, args, full_extraction=True)
        assert images_u_all.shape[0] == args.max_dataset_size, f"images_u_all.shape[0], args.max_dataset_size: {images_u_all.shape[0]}, {args.max_dataset_size}" 
        images_u_all, images_v_all = images_u_all.cuda(), images_v_all.cuda()

        # Calculate true loss
        metrics = get_metrics(args, images_u_all, images_v_all, logit_scale=1)

    else: 
        if args.distributed:
            # should call the set_epoch() method at the beginning of each global_step (for OSGD family)
            feature_loader.sampler.set_epoch(global_step)
        image1_to_image2_, image2_to_image1_ = [], []
        with torch.no_grad():
            for step, ((x_i, x_j), _, idx) in enumerate(tqdm(feature_loader, desc=f'rank[{args.rank}] | feature extraction')):
                x_i = x_i.cuda(args.gpu, non_blocking=True)
                x_j = x_j.cuda(args.gpu, non_blocking=True)

                with torch.cuda.amp.autocast(True):
                    z_i, z_j = model(x_i, x_j, None, None, args, wo_loss=True)
                metrics = get_metrics(args, z_i, z_j, logit_scale=1, tqdm_desc=False)
                image1_to_image2_.append(float(metrics["image1_to_image2_R@1"]))
                image2_to_image1_.append(float(metrics["image2_to_image1_R@1"]))
                x_i, x_j, z_i, z_j = x_i.cpu(), x_j.cpu(), z_i.cpu(), z_j.cpu()
            metrics["image1_to_image2_R@1"] = sum(image1_to_image2_) / len(image1_to_image2_)
            metrics["image2_to_image1_R@1"] = sum(image2_to_image1_) / len(image2_to_image1_)

    # image1:u, image2:v
    image1_to_image2_acc = float(metrics["image1_to_image2_R@1"])
    image2_to_image1_acc = float(metrics["image2_to_image1_R@1"])

    msg = "[{}, {}, epoch={}, N={}, k={}, q={}] u2v={:.4f}, v2u={:.4f}".format(args.data_name, args.pretrained_batch_sampling, args.epoch, args.max_dataset_size, args.k, args.q, image1_to_image2_acc, image2_to_image1_acc)

    # multiple results
    bs_names = (
        f"{args.pretrained_batch_sampling}_u2v_R@1",
        f"{args.pretrained_batch_sampling}_v2u_R@1",
    )
    values = (
        image1_to_image2_acc,
        image2_to_image1_acc,
    )
        

    return args.max_dataset_size, bs_names, values, msg


def evaluation(ckpt_path, epoch, global_args):
    args = get_args(ckpt_path, epoch, global_args)

    model = create_model(epoch, args, global_args)

    train_dataset = get_train_dataset(args, global_args)

    feature_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.feature_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True, \
                collate_fn=train_dataset.collate_fn if global_args.evaluation == "corrupted_top1_accuracy" else None)
    
    if global_args.evaluation == "top1_accuracy": 
        max_dataset_size, bs_name, value, msg = get_top1_accuracy(epoch, args, model, feature_loader, full_batch=True)
    elif global_args.evaluation == "corrupted_top1_accuracy": 
        max_dataset_size, bs_name, value, msg = get_top1_accuracy(epoch, args, model, feature_loader, full_batch=False)
    
    return max_dataset_size, bs_name, value, msg


def main(global_args):

    res_msg = []
    res_dict = defaultdict(lambda:[])
    epoch_list = [-1] + global_args.epoch_list # add random initialized model

    for epoch in epoch_list:
        ckpt_list = global_args.ckpt_paths(epoch)
        res_dict["epochs"].append(epoch+1)
        for ckpt_path in ckpt_list:
            max_dataset_size, bs_name, value, msg = evaluation(ckpt_path, epoch, global_args)   
            res_msg.append(msg)
            if global_args.evaluation in ["top1_accuracy", "corrupted_top1_accuracy"]:
                assert len(bs_name) == len(value)
                for b, v in zip(bs_name, value):
                    res_dict[b].append(v)
            else:
                res_dict[bs_name].append(value)
            if "N" not in res_dict:
                res_dict["N"].append(max_dataset_size)
        res_msg.append("\n")

    if global_args.evaluation in ["top1_accuracy", "corrupted_top1_accuracy"]:
        # save res_msg
        with open(os.path.join(global_args.save_dir, str(global_args.seed), f"{global_args.ckpt_key}_{global_args.evaluation}_{global_args.eval_data_name}_{global_args.data_pathname.replace('/', '-')}_msg.txt"), 'w') as f:
            for line in res_msg:
                f.write(line + '\n')

        # save res_dict
        with open(os.path.join(global_args.save_dir, str(global_args.seed), f"{global_args.ckpt_key}_{global_args.evaluation}_{global_args.eval_data_name}_{global_args.data_pathname.replace('/', '-')}_plot.txt"), 'w') as f:
            for bs_name, losses in res_dict.items():
                f.write(f"{bs_name} = {losses}\n")


if __name__ == "__main__":
    parser = get_parser(description='Evaluation')
    parser.add_argument('--ckpt_key', default='cifar100', type=str,
                    choices=["cifar100_bz32_sim_all", "cifar100_bz32_sog_all", "tiny_imagenet_bz32_sim_all", "tiny_imagenet_bz32_sog_all"])
    parser.add_argument("--epoch_list", nargs='+', type=int, default=[])
    parser.add_argument('--save_dir', metavar='DIR', default='logs/evaluation',
                    help='path to saved results')
    parser.add_argument("--batch_size_plotting", type=int, default=None)
    parser.add_argument('--evaluation', default='corrupted_top1_accuracy', type=str,
                        choices=["top1_accuracy", "corrupted_top1_accuracy"])
    parser.add_argument('--eval_data_name', default=None, type=str)
    parser.add_argument('--data_pathname', default='train', type=str)
    global_args = parser.parse_args()

    global_args.ckpt_paths = {
        "cifar100_bz32_sim_all": lambda epoch:[
            f"logs/cifar100_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sim_bcs_kb_loose_max_k40_q40/checkpoint_{epoch:04d}.pth.tar",
            f"logs/cifar100_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sim_osgd_kb_loose_max_k1500_q150/checkpoint_{epoch:04d}.pth.tar",
            f"logs/cifar100_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sim_s/checkpoint_{epoch:04d}.pth.tar",
            f"logs/cifar100_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sim_sc_even_kb_loose_max_k40_q40/checkpoint_{epoch:04d}.pth.tar",
        ],
        "cifar100_bz32_sog_all": lambda epoch:[
            f"logs/cifar100_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sog_bcs_kb_loose_max_k40_q40/checkpoint_{epoch:04d}.pth.tar",
            f"logs/cifar100_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sog_osgd_kb_loose_max_k1500_q150/checkpoint_{epoch:04d}.pth.tar",
            f"logs/cifar100_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sog_s/checkpoint_{epoch:04d}.pth.tar",
            f"logs/cifar100_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sog_sc_even_kb_loose_max_k40_q40/checkpoint_{epoch:04d}.pth.tar",
        ],
        "tiny_imagenet_bz32_sim_all": lambda epoch:[
            f"logs/tiny_imagenet_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sim_bcs_kb_loose_max_k40_q40/checkpoint_{epoch:04d}.pth.tar",
            f"logs/tiny_imagenet_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sim_osgd_kb_loose_max_k1500_q150/checkpoint_{epoch:04d}.pth.tar",
            f"logs/tiny_imagenet_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sim_s/checkpoint_{epoch:04d}.pth.tar",
            f"logs/tiny_imagenet_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sim_sc_even_kb_loose_max_k40_q40/checkpoint_{epoch:04d}.pth.tar",
        ],
        "tiny_imagenet_bz32_sog_all": lambda epoch:[
            f"logs/tiny_imagenet_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sog_bcs_kb_loose_max_k40_q40/checkpoint_{epoch:04d}.pth.tar",
            f"logs/tiny_imagenet_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sog_osgd_kb_loose_max_k1500_q150/checkpoint_{epoch:04d}.pth.tar",
            f"logs/tiny_imagenet_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sog_s/checkpoint_{epoch:04d}.pth.tar",
            f"logs/tiny_imagenet_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sog_sc_even_kb_loose_max_k40_q40/checkpoint_{epoch:04d}.pth.tar",
        ],
    }[global_args.ckpt_key]

    if global_args.evaluation == "corrupted_top1_accuracy":
        assert len(global_args.data_pathname.split("/")) == 2

    if global_args.seed is None:
        global_args.seed = 42
    set_all_seeds(global_args.seed)
    os.makedirs(os.path.join(global_args.save_dir, str(global_args.seed)), exist_ok=True)

    main(global_args)

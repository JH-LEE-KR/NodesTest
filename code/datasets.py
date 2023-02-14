# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Simple Test Job
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

import utils

def build_dataloader(args):

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    dataset_train, dataset_val = get_dataset(args.dataset, transform_train, transform_val, args)

    args.nb_classes = len(dataset_val.classes)


    if args.distributed and utils.get_world_size() > 1:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    dataloader = {'train': data_loader_train, 'val': data_loader_val}

    return dataloader

def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    else:
        NotImplementedError
    
    return dataset_train, dataset_val

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)
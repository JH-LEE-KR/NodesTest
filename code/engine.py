# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Simple Test Job
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy

import utils

def train_one_epoch(model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, set_training_mode=True, args = None,):

    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)

        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, device, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: '

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)

            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                        device: torch.device, args = None,):

    for epoch in range(args.epochs):            
        _ = train_one_epoch(model=model, criterion=criterion, data_loader=data_loader['train'],
                                    optimizer=optimizer, device=device, epoch=epoch, max_norm=args.clip_grad, 
                                    set_training_mode=True, args=args,)

        test_stats = evaluate(model=model, data_loader=data_loader['val'], device=device, args=args)
        
    result_str = "[Average accuracy]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(test_stats['Acc@1'], test_stats['Acc@5'], test_stats['Loss'])
    print(result_str)

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch,}

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, '{}_test_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')


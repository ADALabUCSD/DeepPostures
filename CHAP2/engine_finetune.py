# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
from datetime import datetime
import torch

from timm.data import Mixup
from timm.utils import accuracy
from torchmetrics.classification import MulticlassRecall, MulticlassSpecificity,MulticlassF1Score
from torchmetrics import ConfusionMatrix
import util.misc as misc
from einops import rearrange

from sklearn.metrics import confusion_matrix
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 1,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, device = torch.device,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.dir))

    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)): 
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = batch[0].float().to(device, non_blocking=True) # BS, 42, 100, 3
        targets = batch[1].to(device, non_blocking=True) # BS,42
        
        batch_size = targets.shape[0] 
        targets = targets.view(-1).squeeze() #(BS*42,)

        
        if True: #criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            targets = targets.float()
        
        if args.model == 'CNNBiLSTMModel':
            samples = rearrange(samples, 'b w l c -> (b w) 1 l c ')
            
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples)
            if args.model == 'CNNBiLSTMModel':
                outputs = outputs.view(-1)
            else:
                outputs = rearrange(outputs, 'b w c -> (b w) c').squeeze(-1)

            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if args.nb_classes == 2:
            preds = torch.round(torch.sigmoid(outputs))
            acc1 = (preds == targets).float().mean()


        else:
            acc1, _ = accuracy(outputs, targets, topk=(1, 2))

        if not math.isfinite(loss_value):
            print('output shape',outputs)
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize() 

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)



        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            #epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.log(
                {'loss': loss_value_reduce, 
                 'lr': max_lr, 
                 'train acc1': acc1,
                #  'train bal_acc':bal_acc,
                #  'train f1':f1,
                 })


            # log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('lr', max_lr, epoch_1000x)
            # log_writer.add_scalar('train acc1', acc1, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args,data_loader, model, device):
    if args.nb_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=args.nb_classes).to(device)
    recall_metric = MulticlassRecall(args.nb_classes,
                            average='weighted',
                            zero_division=0).to(device)
    
    specificity_metric  = MulticlassSpecificity(num_classes=2,
                            average='weighted',
                            zero_division=0).to(device)
    f1_metric = MulticlassF1Score(num_classes=args.nb_classes,
                            average='weighted',
                            zero_division=0).to(device)
    predictions = {'timestamp': [],'prediction': [], 'label': [],'segment': []} 
    for i,batch in enumerate(data_loader):
        samples = batch[0].float().to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        N, W = target.shape
        if True: #criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            target = target.float()
            target = target.view(-1).squeeze()

        if args.model == 'CNNBiLSTMModel': # CHAP
            samples = rearrange(samples, 'b w l c -> (b w) 1 l c ')
            
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples)
            if args.model == 'CNNBiLSTMModel':
                outputs = outputs.view(-1)
            else:
                outputs = rearrange(outputs, 'b w c -> (b w) c').squeeze(-1)

            loss = criterion(outputs, target)
        
        loss_value = loss.item()

        if args.nb_classes == 2:
            preds = torch.round(torch.sigmoid(outputs))
            acc1 = (preds == target).float().mean()
        else:
            acc1, _ = accuracy(outputs, target, topk=(1, 2))
            preds = outputs.argmax(dim=1)     

      
        # record the preds
        if args.make_prediction:
            label_vocabulary = {0: "sitting", 1: "not-sitting", -1: "no-label"}
            
            # prepare timestamp
            timestamp = batch[2] # shape: (BS, 42), need to convert to readable format
            timestamp = timestamp.flatten()
            readable_times = np.vectorize(datetime.fromtimestamp)(timestamp)
            readable_strs = np.vectorize(lambda t: t.strftime("%Y-%m-%d %H:%M:%S"))(readable_times)
            predictions['timestamp'].extend(readable_strs.tolist())

            # prepare pred and target
            str_preds = [label_vocabulary[int(p)] for p in preds]
            str_target = [label_vocabulary[int(t)] for t in target]
            predictions['prediction'].extend(str_preds)
            predictions['label'].extend(str_target)

            # prepare segment id
            segment_id = np.arange(i * N, (i + 1) * N)
            segment_id = np.repeat(segment_id, 42)
            predictions['segment'].extend(segment_id.tolist())


        recall_metric.update(preds, target)
        specificity_metric.update(preds, target)
        f1_metric.update(preds, target)
        confmat_metric.update(preds, target)

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # Compute metrics
    recall_tm = recall_metric.compute().item()
    specificity_tm = specificity_metric.compute().item()
    confmat = confmat_metric.compute()
    bal_acc = 100 * (recall_tm + specificity_tm) / 2
    f1 = 100 * f1_metric.compute().item()

    print('* Acc@1 {top1.global_avg:.5f} bal_acc {bal_acc:.5f} f1 {f1:.5f} loss {losses.global_avg:.3f}'  
          .format(top1=metric_logger.acc1, bal_acc=bal_acc, f1=f1, losses=metric_logger.loss))

    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    eval_stats['f1']=f1
    eval_stats['bal_acc']=bal_acc
    eval_stats['confmat']=confmat
    if args.make_prediction:
        eval_stats['make_prediction'] = predictions

    return eval_stats


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

@torch.no_grad()
def evaluate_cm(data_loader, model, device, as_img=False, save_dir='./'):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    recall_metric = MulticlassRecall(2, average='weighted', zero_division=0).to(device)
    specificity_metric = MulticlassSpecificity(num_classes=2, average='weighted', zero_division=0).to(device)
    f1_metric = MulticlassF1Score(num_classes=2, average='weighted', zero_division=0).to(device)

    all_preds = []
    all_targets = []

    misclassified_series = []
    misclassified_true_labels = []
    misclassified_pred_labels = []

    for samples, target in data_loader:
        samples = samples.float().to(device, non_blocking=True)  # shape: (B, 42, 100, 3)
        original_samples = samples.clone().detach().cpu()

        if as_img:
            samples = rearrange(samples, 'b w l c -> (b w) 1 c l')

        target = target.to(device, non_blocking=True)
        output = model(samples)
        output = output.view(-1, output.size(-1))  # shape: (B*42, C)
        target = target.view(-1)

        loss = criterion(output, target)

        preds = output.argmax(dim=1)

        recall_metric.update(preds, target)
        specificity_metric.update(preds, target)
        f1_metric.update(preds, target)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        acc1, _ = accuracy(output, target, topk=(1, 2))
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        misclassified_mask = preds != target
        misclassified_indices = misclassified_mask.nonzero(as_tuple=True)[0]

        if not as_img:
            for idx in misclassified_indices:
                sample_idx = idx // 42
                time_step_idx = idx % 42
                misclassified_series.append(original_samples[sample_idx, time_step_idx].numpy())
                misclassified_true_labels.append(target[idx].item())
                misclassified_pred_labels.append(preds[idx].item())

    metric_logger.synchronize_between_processes()
    recall_tm = recall_metric.compute().item()
    specificity_tm = specificity_metric.compute().item()
    bal_acc = 100 * (recall_tm + specificity_tm) / 2
    f1 = 100 * f1_metric.compute().item()

    print('* Acc@1 {top1.global_avg:.5f} bal_acc {bal_acc:.5f} f1 {f1:.5f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, bal_acc=bal_acc, f1=f1, losses=metric_logger.loss))

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(2)))

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'misclassified.pkl'), 'wb') as f:
        pickle.dump({
            'series': misclassified_series,
            'true_labels': misclassified_true_labels,
            'pred_labels': misclassified_pred_labels
        }, f)

    return cm, bal_acc

# def evaluate_cm(data_loader, model, device,as_img=False):
#     criterion = torch.nn.CrossEntropyLoss()
#     metric_logger = misc.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     model.eval()

#     recall_metric = MulticlassRecall(2, average='weighted', zero_division=0).to(device)
#     specificity_metric = MulticlassSpecificity(num_classes=2, average='weighted', zero_division=0).to(device)
#     f1_metric = MulticlassF1Score(num_classes=2, average='weighted', zero_division=0).to(device)

#     all_preds = []
#     all_targets = []

#     for samples, target in data_loader:
#         samples = samples.float().to(device, non_blocking=True) # BS,42, 100, 3
#         if as_img:
#             samples = rearrange(samples, 'b w l c -> (b w) 1 c l')
#             #samples = samples.unsqueeze(1)
#         target = target.to(device, non_blocking=True)

#         output = model(samples)
#         output = output.view(-1, output.size(-1))  # (BS * 42, C)
#         target = target.view(-1)

#         loss = criterion(output, target)

#         preds = output.argmax(dim=1)

#         recall_metric.update(preds, target)
#         specificity_metric.update(preds, target)
#         f1_metric.update(preds, target)

#         all_preds.extend(preds.cpu().numpy())
#         all_targets.extend(target.cpu().numpy())

#         acc1, _ = accuracy(output, target, topk=(1, 2))
#         batch_size = samples.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

#     metric_logger.synchronize_between_processes()
#     recall_tm = recall_metric.compute().item()
#     specificity_tm = specificity_metric.compute().item()
#     bal_acc = 100 * (recall_tm + specificity_tm) / 2
#     f1 = 100 * f1_metric.compute().item()

#     print('* Acc@1 {top1.global_avg:.5f} bal_acc {bal_acc:.5f} f1 {f1:.5f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, bal_acc=bal_acc, f1=f1, losses=metric_logger.loss))

#     cm = confusion_matrix(all_targets, all_preds, labels=list(range(2)))
    
#     return cm,bal_acc


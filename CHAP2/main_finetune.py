# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime

import os
import time
from pathlib import Path
import copy
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import wandb
# from timm.scheduler.cosine_lr import CosineLRScheduler
import seaborn as sns
import matplotlib.pyplot as plt


# from config import FT_LONG_DATASET_CONFIG
from util.datasets import data_aug, iWatch 
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from timm.optim import create_optimizer_v2
from engine_finetune import train_one_epoch, evaluate
from util.loss import BinaryFocalLoss
import pandas as pd

import pickle


from tqdm import tqdm

from chap_model import CHAP,CNNAttentionModel,FeatureExtractorWrapper
from util.chap_utils import load_model_weights
from util.commons import get_subjectwise_dataloaders
from omegaconf import OmegaConf

def get_args_parser():
    parser = argparse.ArgumentParser('CHAP-FT', add_help=False)
    parser.add_argument('--config', default=None, type=str,
                        help='path to config file (default: None, use default config)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--make_prediction', action='store_true',)
    parser.add_argument('--prediction_dir', default=None, type=str,
                        help='Directory to save prediction files') 

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', type=int, default=4200, 
                        help='Input size "')
    parser.add_argument('--patch_size', type=int, default=100, 
                        help='Patch size')
    parser.add_argument('--patch_nvar', type=int, default=1, 
                    help='Patch size')
    parser.add_argument('--use_pos_embed', action='store_true', default=False,)
    parser.add_argument('--no_use_pos_embed', action='store_false', dest='use_pos_embed',)

    parser.add_argument('--in_chans', default=3, type=int,  # changed - added
                        help='number of channels')
    parser.add_argument('--remark', default='Debug',type=str,
                        help='model_remark')
    parser.add_argument('--use_data_aug',default=1,type=int)
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='Drop path rate')
    parser.add_argument('--patch_emb', type=str, default='vit', #sundial
                        help='Patch embedding type')
    parser.add_argument('--use_rope', action='store_true',
                        help='Use rotary position embedding')
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=5e-2,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--num_attn_layer', type=int, default=2,
                        help='number of attention layers in the AttentionProbeModel')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR', # default 1e-2
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--learnable_pos_embed', action='store_true', default=False,
                        help='use learnable position embedding (default: False)')
    parser.add_argument('--no_learnable_pos_embed', action='store_false', dest='learnable_pos_embed',
                        help='disable learnable position embedding')

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--pos_weight', type=float, default=1.0, 
                        help='positive weight for BCE loss')
    parser.add_argument('--use_focal_loss', action='store_true',
                    help='Use focal loss instead of BCEWithLogitsLoss')
    parser.add_argument('--subset_ratio',type=float,default=1.0,
                        help='Subset ratio for the dataset, default is 1.0 (use all data)')
    # * Finetuning params
    parser.add_argument('--checkpoint', default=None, 
                        type=str,help='model checkpoint for evaluation') 
                        

    # Dataset parameters
    parser.add_argument('--data_path', default='/niddk-data-central/iWatch/pre_processed_seg/W', type=str, # changed
                        help='dataset path')
    
    parser.add_argument('--nb_classes', default=2, type=int, # changed
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='/niddk-data-central/ckpt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/niddk-data-central/log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=None, type=str,
                        help='Perform evaluation only')
    parser.add_argument('--subject_level_analysis', action='store_true',)
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    return parser    



def main(args):
    args_dict = vars(args)
    if args.config:
        cfg = OmegaConf.load(args.config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        flat_cfg = misc.flatten_config_dict(cfg_dict)  # flatten the config dict
    else:
        flat_cfg = {}

    combined_config = {**args_dict, **flat_cfg}
    

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False 
    print('Using iWatch HDF5 dataloader')
    transform=None
    if args.use_data_aug:
        transform=data_aug
    
    dataset_train = iWatch(
        set_type='train',
        root=args.data_path,
        transform=transform,
        subset_ratio=args.subset_ratio,)
    dataset_val = iWatch(
        set_type='val',
        root=args.data_path,
        transform=None,)
    dataset_test = iWatch(
        set_type='test_complete',
        root=args.data_path,
        transform=None,)

    print(f"using {args.subset_ratio} of train dataset, {len(dataset_train)} samples")

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=int(args.num_workers//2),
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,  )

    if args.log_dir is not None and not args.eval and global_rank == 0:  
        wandb.login(key=WANDB_KEY)
        log_writer = wandb.init(
            project='CHAP_FT',  # Specify your project
            config= combined_config,
            dir=args.log_dir,
            name=args.remark,)
      
    else:
        log_writer = None

    # TODO: package below into a model factory function
    # CHAP replicate #######
    if args.model == 'CHAP':
        model = CHAP(2,42,2)

        if args.checkpoint:
            msg = load_model_weights(model, args.checkpoint, weights_only=False)
        else:
            print("training CHAP from scratch")
        
    
    elif args.model == 'CNNBiLSTMAttentionModel':
        # add attention on top of CNNBiLSTM
        base_model = CHAP(2,42,2)

        if args.checkpoint:
            msg = load_model_weights(model, args.checkpoint, weights_only=False)
        else:
            print("training CHAP from scratch")


        base_model_hidden_dim = base_model.fc_bilstm.in_features # 256
        print("base_model_hidden_dim:", base_model_hidden_dim)
        base_model = FeatureExtractorWrapper(base_model)
        model = CNNAttentionModel(base_model=base_model,
                                          base_model_hidden_dim=base_model_hidden_dim,
                                          num_layer=cfg.model.num_layers,
                                          hidden_dim=cfg.model.hidden_dim,
                                          num_heads=cfg.model.num_heads,
                                          ffn_multiplier=cfg.model.ffn_multiplier,
                                          drop_path_rate=cfg.model.drop_path_rate,
                                          learnable_pos_embed=cfg.model.learnable_pos_embed,)

    elif args.model == 'CNNAttentionModel':
        base_model = CHAP(2,42,2)
        if cfg.model.transfer_learning_model_path:
            msg = load_model_weights(base_model, cfg.model.transfer_learning_model_path, weights_only=False)
            print(msg)
        
        # we only need the CNN extractor
        base_model = base_model.cnn_model

        base_model_hidden_dim = base_model.fc.out_features # 512
        model = CNNAttentionModel(base_model=base_model,
                                          base_model_hidden_dim=base_model_hidden_dim,
                                          num_layer=cfg.model.num_layers,
                                          hidden_dim=cfg.model.hidden_dim,
                                          num_heads=cfg.model.num_heads,
                                          ffn_multiplier=cfg.model.ffn_multiplier,
                                          drop_path_rate=cfg.model.drop_path_rate,
                                          learnable_pos_embed=cfg.model.learnable_pos_embed,)
                                    
        
    #######################
    elif args.model == 'vit-base':
        base_model = MaskedAutoencoderViT(img_size=[3,args.input_size],
                                patch_size=[args.patch_nvar,args.patch_size],
                                patch_emb=args.patch_emb,
                                use_rope = args.use_rope,
                                learnable_pos_embed=args.learnable_pos_embed,
                                embed_dim=768,
                                depth=12,
                                num_heads=12)
        
        # TODO: No weight to load right now.
        model = ClassiferHeadWrapper(base_model, num_classes=args.nb_classes)

    elif args.model == 'vit-small':
        base_model = MaskedAutoencoderViT(img_size=[3,args.input_size],
                                patch_size=[args.patch_nvar,args.patch_size],
                                patch_emb=args.patch_emb,
                                use_rope = args.use_rope,
                                learnable_pos_embed=args.learnable_pos_embed,
                                embed_dim=384,
                                depth=12,
                                num_heads=6)
        
        # TODO: No weight to load right now.
        model = ClassiferHeadWrapper(base_model, num_classes=args.nb_classes)

    elif args.model == 'vit-tiny':
        base_model = MaskedAutoencoderViT(img_size=[3,args.input_size],
                                patch_size=[args.patch_nvar,args.patch_size],
                                patch_emb=args.patch_emb,
                                use_rope = args.use_rope,
                                learnable_pos_embed=args.learnable_pos_embed,
                                embed_dim=192,
                                depth=12,
                                num_heads=3)
        
        # TODO: No weight to load right now.
        model = ClassiferHeadWrapper(base_model, num_classes=args.nb_classes)


    if args.eval:
        # Evaluate 
        checkpoint = torch.load(args.eval,map_location='cpu',weights_only=False)
        try:
            checkpoint_model = checkpoint['model']
            print(checkpoint['args'])
        except KeyError:
            checkpoint_model = checkpoint

        
        msg = model.load_state_dict(checkpoint_model, strict=False)
        # only pos_embed can be missed
        # assert 'pos_embed' in msg.missing_keys or 'pos_embed' in msg.unexpected_keys, \
        #     f"Unexpected keys: {msg.unexpected_keys}, Missing keys: {msg.missing_keys}"

        print(msg)
        model.to(device)
        print(model)

        if args.subject_level_analysis:
            train_subject_list = list(dataset_train.subject_id)
            val_subject_list = list(dataset_val.subject_id)

            subject_performance={'train':{},'val':{}}
            train_subject_dataloader = get_subjectwise_dataloaders(dataset_train,batch_size=args.batch_size)
            val_subject_dataloader = get_subjectwise_dataloaders(dataset_val,batch_size=args.batch_size) 
            
            for subject_id in tqdm(train_subject_list):
                subject_id = subject_id.decode("utf-8") if isinstance(subject_id, bytes) else subject_id
                print(f"Evaluating subject {subject_id} in train set")
                train_stats = evaluate(args,train_subject_dataloader[subject_id], model, device)
                print(f"Balanced Accuracy on subject {subject_id}: {train_stats['bal_acc']:.5f}% and F1 score of {train_stats['f1']:.5f}%")
                subject_performance['train'][subject_id] = {}
                subject_performance['train'][subject_id]['bal_acc'] = train_stats['bal_acc']
                subject_performance['train'][subject_id]['f1'] = train_stats['f1']
            
            for subject_id in tqdm(val_subject_list):
                subject_id = subject_id.decode("utf-8") if isinstance(subject_id, bytes) else subject_id
                print(f"Evaluating subject {subject_id} in validation set")
                test_stats = evaluate(args,val_subject_dataloader[subject_id], model, device)
                print(f"Balanced Accuracy on subject {subject_id}: {test_stats['bal_acc']:.5f}% and F1 score of {test_stats['f1']:.5f}%")
                subject_performance['val'][subject_id] = {}
                subject_performance['val'][subject_id]['bal_acc'] = test_stats['bal_acc']
                subject_performance['val'][subject_id]['f1'] = test_stats['f1']

            # save subject_performance in loacl folder
            output_dir = os.path.join('subject_level_performance',args.model,f'{args.remark}_subject_performance.pkl')
            
            Path(os.path.dirname(output_dir)).mkdir(parents=True, exist_ok=True)
            with open(output_dir, 'wb') as f:
                pickle.dump(subject_performance, f)
            
            exit(0)

        else:
            val_stats = evaluate(args,data_loader_val, model, device)
            print(f"Balanced Accuracy of the network in validation-set: {val_stats['bal_acc']:.5f}% and F1 score of {val_stats['f1']:.5f}%")
            test_stats = evaluate(args,data_loader_test, model, device)
            print(f"Balanced Accuracy of the network in test-set: {test_stats['bal_acc']:.5f}% and F1 score of {test_stats['f1']:.5f}%")
            train_stats = evaluate(args,data_loader_train, model, device)
            print(f"Balanced Accuracy of the network in training-set: {train_stats['bal_acc']:.5f}% and F1 score of {train_stats['f1']:.5f}%")

            # Create directory for prediction_dir
            ckpt_path = os.path.join(args.prediction_dir, args.model, 'checkpoint')
            prediction_file_root = os.path.join(args.prediction_dir, args.model, 'predictions')
            os.makedirs(ckpt_path, exist_ok=True)
            os.makedirs(prediction_file_root, exist_ok=True)

            # Copy checkpoint file if it exists  
            dst_path = os.path.join(ckpt_path, 'checkpoint-submit.pth')
            shutil.copy(args.eval, dst_path)
            print(f"Copying checkpoint from {args.eval} to {dst_path}")
    

            if args.make_prediction:
                test_subject_dataloader = get_subjectwise_dataloaders(dataset_val,batch_size=args.batch_size) 
                test_subject_list = list(dataset_val.subject_id)
                subject_performance={'test':{}}
                
                for subject_id in  tqdm(test_subject_list):
                    subject_id = subject_id.decode("utf-8") if isinstance(subject_id, bytes) else subject_id
                    prediction_file = os.path.join(prediction_file_root,f'{subject_id}.csv')

                    print(f"Evaluating subject {subject_id} in test set")
                    test_stats = evaluate(args,test_subject_dataloader[subject_id], model, device)
                    print(f"Balanced Accuracy on subject {subject_id}: {test_stats['bal_acc']:.5f}% and F1 score of {test_stats['f1']:.5f}%")
                    subject_performance['test'][subject_id] = {}
                    subject_performance['test'][subject_id]['bal_acc'] = test_stats['bal_acc']
                    subject_performance['test'][subject_id]['f1'] = test_stats['f1']
                    
                    # write prediction to csv file
                    prediction = test_stats['make_prediction'] 
                    if prediction is not None:
                        segment = prediction['segment']
                        timestamp = prediction['timestamp']
                        pred = prediction['prediction']
                        label = prediction['label']
                        df = pd.DataFrame({'segment': segment, 'timestamp': timestamp, 'prediction': pred, 'label': label})
                        df.to_csv(prediction_file, index=False)

            exit(0)

    print("Model = %s" % str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of training params : %.2f' % (n_parameters))

    n_total_parameters = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %.2f' % (n_total_parameters))

    # Print non-trainable layers
    print("\nLayers with requires_grad=False:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f" - {name}: shape={tuple(param.shape)}")
            
    model.to(device)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    model_without_ddp = model
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    print("lr: %.3e" % args.lr)

    if args.distributed: #changed - hashed out
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module
    
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(f"[DDP UNUSED PARAM] {name}")

    if args.model == 'CHAP':
        optimizer = create_optimizer_v2(
            model_without_ddp,
            opt='adamw',
            lr=args.lr,
            weight_decay=args.weight_decay,) # default: 0 )
    else:
        # add layer decay
        optimizer = create_optimizer_v2(
        model_without_ddp,
        opt='adamw',
        lr=args.lr,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,)

    loss_scaler = NativeScaler()

    if args.use_focal_loss:
        criterion = BinaryFocalLoss(alpha=0.25,gamma=2.0, reduction="mean").to(device) # alpha is for balance, gamma is to ignore easy sample
    elif args.nb_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], dtype=torch.float32).to(device))
        
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    #if not args.CHAP:
    # scheduler = CosineLRScheduler(
    # optimizer,
    # t_initial=args.epochs,
    # warmup_t=args.warmup_epochs,
    # warmup_lr_init=args.min_lr,
    # t_in_epochs=True)
    # else:
    #     scheduler = None

    print("criterion = %s" % str(criterion))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_metric = {'epoch': 0, 'acc1': 0.0, 'bal_acc': 0.0, 'f1': 0.0}
    save_buffer = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed: 
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, loss_scaler,
            max_norm=args.clip_grad, 
            log_writer=log_writer,
            args=args, device=device,
        )


        test_stats = evaluate(args, data_loader_val, model, device)
        print(f"Balanced Accuracy of the network on test images: {test_stats['bal_acc']:.5f} and F1 score of {test_stats['f1']:.5f}%")

        # Avoid NCCL Comm error
        if torch.distributed.is_initialized():
            print('Watiing for all processes to finish')
            torch.distributed.barrier()

        if max_accuracy < test_stats["bal_acc"]:
            max_accuracy = test_stats["bal_acc"]

            best_metric['epoch'] = epoch
            best_metric['bal_acc'] = test_stats['bal_acc']
            best_metric['acc1'] = test_stats['acc1']
            best_metric['f1'] = test_stats['f1']
            best_metric['confmat'] = test_stats['confmat']

            # Save best state_dicts
            save_buffer = {
                'model': copy.deepcopy(model.state_dict()),
                'model_without_ddp': copy.deepcopy(model_without_ddp.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'loss_scaler': copy.deepcopy(loss_scaler.state_dict())
            }

        if log_writer is not None:
            log_writer.log({
                'perf/test_acc1': test_stats['acc1'], 
                'perf/bal_acc': test_stats['bal_acc'],
                'perf/f1': test_stats['f1'],
                'perf/test_loss': test_stats['loss'], 
                'epoch': epoch
            })

    # Save final model
    if args.output_dir:
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        if save_buffer is not None:
            print(f"Saving best model from epoch {best_metric['epoch']} with balanced accuracy {best_metric['bal_acc']:.4f}")

            # Reload state dicts into model before saving
            model.load_state_dict(save_buffer['model'])
            model_without_ddp.load_state_dict(save_buffer['model_without_ddp'])
            optimizer.load_state_dict(save_buffer['optimizer'])
            loss_scaler.load_state_dict(save_buffer['loss_scaler'])

            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch='best')

    # Logging confusion matrix
    if log_writer is not None:
        confmat = best_metric['confmat'].cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['sitting', 'non-sitting'], yticklabels=['sitting', 'non-sitting'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        log_writer.log({
            "best_epoch_bal_acc": best_metric['bal_acc'],
            "best_epoch_acc1": best_metric['acc1'],
            "best_epoch_f1": best_metric['f1'],
            "best_epoch": best_metric['epoch'],
            "best_epoch_confmat": wandb.Image(fig)
        })

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    return max_accuracy


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    initial_timestamp = datetime.datetime.now()
    
    args.remark = args.remark + f'set_{args.subset_ratio}_blr_{args.blr}_bs_{args.batch_size}_input_size_{args.input_size}'
    print(f'Start Training: {args.remark}')
    
    args.log_dir = os.path.join(args.log_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    args.output_dir = os.path.join(args.output_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    if args.output_dir and not args.eval:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)



'''
 
## SOL ########################################

# CHAP-FT

torchrun --nproc_per_node=4 -m main_finetune_long \
--data_path "/niddk-data-central/SOL/PASOS/train/SOL_10hz" \
--remark CHAP-FT  \
--blr 1e-3 \
--model CHAP \
--checkpoint "MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth" \
--epochs 10 \
--warmup_epochs 2 \
--batch_size 16 \
--weight_decay 1e-3 \
--subset_ratio 1.0 \
--pos_weight 1.0 \
--use_data_aug 1 



'''
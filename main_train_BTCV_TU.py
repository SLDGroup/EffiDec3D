#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat June  21 11:06:19 2025

@author: mostafij.rahman
python main_finetune_BTCV_TU.py --root /research/data/amos_trns/ --output output_folder/run1 --dataset BTCV13 --img_size 96 96 96 --n_channels 1 --network 3DUXNET_EffiDec3D --channels 48 96 192 384 --n_decoder_channels 48 --ds False --mode train --pretrain False --batch_size 1 --crop_sample 4 --lr 0.001 --optim AdamW --max_iter 45000  --eval_step 250 --val_batch 1 --gpu 0 --cache_rate 1.0 --num_workers 4 --overlap 0.7 > output_folder/BTCV13_3DUXNET_EffiDec3D_loss_dsFalse_1out_96x96x96_lr1e3_itr45000_overlap070_run1.txt

"""

from torch.cuda.amp import autocast, GradScaler

from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.networks.nets import UNETR, SwinUNETR
from networks.swin_unetr_effidec3d import SwinUNETR as SwinUNETRv2
from networks.MedNeXt.mednextv1.create_mednext_v1 import create_mednext_v1
from networks.UXNet_3D.network_backbone import UXNET, UXNET_EffiDec3D
from networks.swin_unetr_effidec3d import SwinUNETR_EffiDec3D
from networks.MedNeXt.mednextv1.create_mednextv1_effidec3d import create_mednextv1_effidec3d

from networks.unetr_pp.synapse.unetr_pp_synapse import UNETR_PP
from networks.nnunet.network_architecture.generic_UNet import Generic_UNet
from networks.SegFormer3D.segformer3d import SegFormer3D 
from networks.SlimUNETR.SlimUNETR import SlimUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceCELoss
from monai_utils.inferers.utils import sliding_window_inference_1out
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    Activations,
    )

import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms
from monai.networks.blocks import UnetOutBlock
import torch.nn as nn
import torch.nn.functional as F

from ptflops import get_model_complexity_info

import csv
import os
import numpy as np
import scipy.ndimage as ndimage
from medpy import metric
from tqdm import tqdm
import argparse

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser(description='3DUXNET w/ EffiDec3D hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='data', required=True, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='', required=True, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=True, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
parser.add_argument('--img_size', type=int, nargs='+', default=[96,96,96], help='3D ROI size, e.g., [96, 96, 96]') #500
parser.add_argument('--n_channels', type=int, default=1, help='number of channels in input image') #500

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='3DUXNET_EffiDec3D', help='Network models: {3DUXNET_EffiDec3D, SwinUNETR_EffiDec3D, SwinUNETRv2_EffiDec3D, MedNeXt_M_EffiDec3D, TransBTS, nnFormer, UNETR, SlimUNETR, SwinUNETR, 3DUXNET, MedNeXt_M}')
parser.add_argument('--channels', type=int, nargs='+', default=[48, 96, 192, 384], help='Number of channels in the 3DUXNet network, e.g., [48, 96, 192, 384]') #500
parser.add_argument('--feature_size', type=int, default=48, help='Feature size for SwinTransformer') #500
parser.add_argument('--n_decoder_channels', type=str, default="48", help='Number of channels in each satge of the decoder') #500
parser.add_argument('--resolution_factor', type=int, default=2, help='Resolution factor to control high-resolution operations, e.g., [1,2,4,8,16]') #500
parser.add_argument('--skip_aggregation', type=str, default='addition', required=False, help='Aggregation in skip connection: {addition, concatenation}')

parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--ds', default=True, help='Use of deep supervision (ds) or not')
parser.add_argument('--pretrained_weights', default='', help='Path of pretrained weights')
parser.add_argument('--pretrain_classes', default='', help='Number of classes output from pretrained model')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=50, help='Per steps to perform validation') #500
parser.add_argument('--val_batch', type=int, default=1, help='Validation batch size') #500
parser.add_argument('--overlap', type=float, default=0.5, help='Amount of overlap between scans') #500
parser.add_argument('--overlap_mode', type=str, default='constant', help='overlap mode') #500

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

train_samples, valid_samples, out_classes = data_loader(args)

train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
]

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]

print(train_files)
print(val_files)

set_determinism(seed=0)

train_transforms, val_transforms = data_transforms(args)

## Train Pytorch Data Loader and Caching
print('Start caching datasets!')
train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=args.cache_rate, num_workers=args.num_workers)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

## Valid Pytorch Data Loader and Caching
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

if args.n_decoder_channels == 'None':
    args.n_decoder_channels = None
else:
    args.n_decoder_channels = int(args.n_decoder_channels)
if args.ds == 'True':
    args.ds = True
    
## Load Networks
device = torch.device("cuda:0")

if args.network == '3DUXNET_EffiDec3D':
	model = UXNET_EffiDec3D(
        in_chans=args.n_channels, #1
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=args.channels,#[48, 96, 192, 384],
        n_decoder_channels=args.n_decoder_channels,
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
        skip_aggregation=args.skip_aggregation,
        resolution_factor=args.resolution_factor
    ).to(device)
    
elif args.network == 'SwinUNETR_EffiDec3D':
	model = SwinUNETR_EffiDec3D(
        img_size=args.img_size,
        in_channels=args.n_channels, #1
        out_channels=out_classes,
        feature_size=args.feature_size,
        n_decoder_channels=args.n_decoder_channels,
        resolution_factor=args.resolution_factor,
        use_checkpoint=False,
        skip_aggregation=args.skip_aggregation,
        use_v2=False
   	).to(device)

elif args.network == 'SwinUNETRv2_EffiDec3D':
	model = SwinUNETR_EffiDec3D(
        img_size=args.img_size,
        in_channels=args.n_channels, #1
        out_channels=out_classes,
        feature_size=args.feature_size,
        n_decoder_channels=args.n_decoder_channels,
        resolution_factor=args.resolution_factor,
        use_checkpoint=False,
        skip_aggregation=args.skip_aggregation,
        use_v2=True
   	).to(device)

elif args.network == 'MedNeXt_M_EffiDec3D':
    model = create_mednextv1_effidec3d(
        args.n_channels, 
        out_classes, 
        'M', 
        n_channels=args.feature_size,
        kernel_size=3,
        deep_supervision=args.ds #True
    ).to(device)

## 3D UX-Net
elif args.network == '3DUXNET':
    #print(args.pretrain)
    if args.pretrain == True:
        #print('here')
        model = UXNET(
            in_chans=args.n_channels,
            out_chans=args.pretrain_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        )
        model.load_state_dict(torch.load(args.pretrained_weights))
        model.out = UnetOutBlock(spatial_dims=3, in_channels=48, out_channels=out_classes)
        model = model.to(device)
    else:
        model = UXNET(
            in_chans=args.n_channels, #1
            out_chans=out_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        ).to(device)

elif args.network == 'SlimUNETR':
    if args.img_size[0] == 96:
        embedding_dim = 27
    else:
        embedding_dim = 64
    
    model = SlimUNETR(
        in_channels=args.n_channels,
        out_channels=out_classes,
        embed_dim=96,
        embedding_dim=embedding_dim,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ).to(device)
            
## SwinUNETR
elif args.network == 'SwinUNETR':
    if args.pretrain == True:
        model = SwinUNETR(
            img_size=args.img_size,
            in_channels=args.n_channels,
            out_channels=args.pretrain_classes,
            feature_size=args.feature_size,
            use_checkpoint=False,
        )
        model.load_state_dict(torch.load(args.pretrained_weights))
        model.out = UnetOutBlock(spatial_dims=3, in_channels=48, out_channels=out_classes)
        model = model.to(device)
    else:
        model = SwinUNETR(
            img_size=args.img_size,
            in_channels=args.n_channels, #1
            out_channels=out_classes,
            feature_size=args.feature_size, #48
            use_checkpoint=False,
        ).to(device)

elif args.network == 'SwinUNETRv2':
    model = SwinUNETRv2(
            img_size=args.img_size,
            in_channels=args.n_channels, #1
            out_channels=out_classes,
            feature_size=args.feature_size, #48
            use_checkpoint=False,
            use_v2=True
        ).to(device)

elif args.network == 'nnUNet':
    model = Generic_UNet(
        input_channels=args.n_channels, 
        base_num_features=48, 
        num_classes=out_classes, 
        num_pool=4, 
        num_conv_per_stage=2,
        conv_op=nn.Conv3d,
        norm_op=nn.BatchNorm3d,
        dropout_op=nn.Dropout3d,
        max_num_features=512,
        deep_supervision=False,
    ).to(device)        
        
## nnFormer
elif args.network == 'nnFormer':
    if args.pretrain == True:
        from networks.nnFormer.nnFormer_seg import final_patch_expanding
        final_layer = []
        model = nnFormer(input_channels=args.n_channels, num_classes=args.pretrain_classes)
        model.load_state_dict(torch.load(args.pretrained_weights))
        final_layer.append(final_patch_expanding(192, out_classes, patch_size=[2,4,4]))
        model.final = nn.ModuleList(final_layer)
        model = model.to(device)
    else:
        model = nnFormer(input_channels=args.n_channels, num_classes=out_classes).to(device) #1

## UNETR
elif args.network == 'UNETR':
    if args.pretrain == True:
        model = UNETR(
            in_channels=args.n_channels,
            out_channels=args.pretrain_classes,
            img_size=args.img_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
        model.load_state_dict(torch.load(args.pretrained_weights))
        model.out = UnetOutBlock(spatial_dims=3, in_channels=48, out_channels=out_classes)
        model = model.to(device)
    else:
        model = UNETR(
            in_channels=args.n_channels, #1
            out_channels=out_classes,
            img_size=args.img_size,
            feature_size=args.feature_size, #16
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)
        
elif args.network == 'UNETR_PP':
    model = UNETR_PP(
        in_channels=args.n_channels, #1
        out_channels=out_classes,
        img_size=args.img_size,
        feature_size=args.feature_size, #16
        hidden_size=256,
        dims=[32, 64, 128, 256],
        num_heads=4,
        pos_embed="perceptron",
        norm_name="instance",
        dropout_rate=0.0,
        do_ds=False
    ).to(device)

elif args.network == 'MedNeXt_S':
    model = create_mednext_v1(
        args.n_channels, 
        out_classes, 
        'S', 
        kernel_size=5,
        n_channels=args.feature_size,
        deep_supervision=args.ds #False
    ).to(device)
    
elif args.network == 'MedNeXt_B':
    model = create_mednext_v1(
        args.n_channels, 
        out_classes, 
        'B', 
        kernel_size=5,
        n_channels=args.feature_size,
        deep_supervision=args.ds #False
    ).to(device)

elif args.network == 'MedNeXt_M':
    model = create_mednext_v1(
        args.n_channels, 
        out_classes, 
        'M', 
        kernel_size=5,
        n_channels=args.feature_size,
        deep_supervision=args.ds #False
    ).to(device)

elif args.network == 'MedNeXt_L':
    model = create_mednext_v1(
        args.n_channels, 
        out_classes, 
        'L', 
        kernel_size=5,
        n_channels=args.feature_size,
        deep_supervision=args.ds #False
    ).to(device)

elif args.network == 'SegFormer3D':
    model = SegFormer3D(
        in_channels = args.n_channels,
        sr_ratios = [4, 2, 1, 1],
        embed_dims = [32, 64, 160, 256],
        patch_kernel_size = [7, 3, 3, 3],
        patch_stride = [4, 2, 2, 2],
        patch_padding = [3, 1, 1, 1],
        mlp_ratios = [4, 4, 4, 4],
        num_heads = [1, 2, 5, 8],
        depths = [2, 2, 2, 2],
        decoder_head_embedding_dim = 256,
        num_classes = out_classes,
        decoder_dropout = 0.0,
    ).to(device)
    
## TransBTS
elif args.network == 'TransBTS':
    if args.pretrain == True:
        #_, model = TransBTS(dataset='flare', _conv_repr=True, _pe_type='learned')
        _, model = TransBTS(num_classes=2, num_channels=1, img_dim=96, _conv_repr=True, _pe_type='learned')
        model.load_state_dict(torch.load(args.pretrained_weights))
        model.endconv = nn.Conv3d(512 // 32, out_classes, kernel_size=1)
        model = model.to(device)
    else:
        #_, model = TransBTS(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
        _, model = TransBTS(num_classes=out_classes, num_channels=args.n_channels, img_dim=args.img_size[0], _conv_repr=True, _pe_type='learned')
        
        model = model.to(device)

print('Chosen Network Architecture: {}'.format(args.network))

macs, params = get_model_complexity_info(model, (args.n_channels, args.img_size[0], args.img_size[1], args.img_size[2]), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    model = nn.DataParallel(model)
#model.to(device)
                                            
## Define Loss function and optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))

if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)
scaler = GradScaler()

if '3DUXNET' in args.network and 'EffiDec3D' in args.network:
    args.output = args.output + '_network_' + args.network + '_fc' + str(args.channels[0]) + '_' + str(args.channels[1]) + '_' + str(args.channels[2]) + '_' + str(args.channels[3]) + '_n_decoder_channels' + str(args.n_decoder_channels) + '_rf'+str(args.resolution_factor)+'_skip_aggregation_'+args.skip_aggregation + '_roi' + str(args.img_size[0]) + 'x' + str(args.img_size[1]) + 'x' + str(args.img_size[2])+'_vb'+str(args.val_batch)+'_cs'+str(args.crop_sample)+'_overlap'+str(args.overlap) + '_ds' + str(args.ds) + '_pretrain_'+str(args.pretrain)+'_lr_'+str(args.lr)
elif 'EffiDec3D' in args.network:
    args.output = args.output + '_network_' + args.network + '_fs' + str(args.feature_size) + '_n_decoder_channels' + str(args.n_decoder_channels) + '_rf'+str(args.resolution_factor)+'_skip_aggregation_'+args.skip_aggregation + '_roi' + str(args.img_size[0]) + 'x' + str(args.img_size[1]) + 'x' + str(args.img_size[2])+'_vb'+str(args.val_batch)+'_cs'+str(args.crop_sample)+'_overlap'+str(args.overlap) + '_ds' + str(args.ds) + '_pretrain_'+str(args.pretrain)+'_lr_'+str(args.lr)
elif '3DUXNET' in args.network:
    args.output = args.output + '_network_' + args.network + '_fc' + str(args.channels[0]) + '_' + str(args.channels[1]) + '_' + str(args.channels[2]) + '_' + str(args.channels[3]) + '_roi' + str(args.img_size[0]) + 'x' + str(args.img_size[1]) + 'x' + str(args.img_size[2])+'_vb'+str(args.val_batch)+'_cs'+str(args.crop_sample)+'_overlap'+str(args.overlap) + '_ds' + str(args.ds) + '_pretrain_'+str(args.pretrain)+'_lr_'+str(args.lr)
else:
    args.output = args.output + '_network_' + args.network + '_fs' + str(args.feature_size) + '_roi' + str(args.img_size[0]) + 'x' + str(args.img_size[1]) + 'x' + str(args.img_size[2])+'_vb'+str(args.val_batch)+'_cs'+str(args.crop_sample)+'_overlap'+str(args.overlap) + '_ds' + str(args.ds) + '_pretrain_'+str(args.pretrain)+'_lr_'+str(args.lr)

root_dir = os.path.join(args.output,args.network,args.dataset)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)

last_ckpt = os.path.join(root_dir, "last_model.pth")

ckpt = None
if os.path.isfile(last_ckpt):
    size = os.path.getsize(last_ckpt)
    if size > 0:
        try:
            ckpt = torch.load(last_ckpt, map_location=device)
            print(f"=> Loaded checkpoint (step {ckpt.get('global_step', '?')})")
        except EOFError:
            print(f"[WARN] Checkpoint file {last_ckpt!r} is corrupted (EOF). Removing it.")
            os.remove(last_ckpt)
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e!r}. Removing it.")
            os.remove(last_ckpt)
    else:
        print(f"[WARN] Checkpoint file {last_ckpt!r} is empty. Removing it.")
        os.remove(last_ckpt)

# if ckpt was loaded, restore; otherwise default init
if ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    global_step      = ckpt.get("global_step", 0)
    dice_val_best    = ckpt.get("dice_val_best", 0.0)
    global_step_best = ckpt.get("global_step_best", 0)
else:
    print("=> No valid checkpoint, training from scratch")
    global_step = 0

t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            if out_classes == 9:
                val_labels[val_labels==5] = 0
                val_labels[val_labels==9] = 0
                val_labels[val_labels==10] = 0
                val_labels[val_labels==12] = 0
                val_labels[val_labels==13] = 0
                val_labels[val_labels==11] = 5
            
            with autocast(enabled=False):
                val_outputs = sliding_window_inference_1out(val_inputs, (args.img_size[0], args.img_size[1], args.img_size[2]), args.val_batch, model, overlap=args.overlap)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f mean_dice=%2.5f)" % (global_step, 10.0, dice, np.mean(dice_vals))
                )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    writer.add_scalar('Validation Segmentation Loss', mean_dice_val, global_step)
    return mean_dice_val

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
        
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with autocast(enabled=False):
            p = model(x, mode='train')
            P = []
            if type(p) is not list:
                P.append(p)
            else:
                P = p
            if out_classes == 9:       
                y[y==5] = 0
                y[y==9] = 0
                y[y==10] = 0
                y[y==12] = 0
                y[y==13] = 0
                y[y==11] = 5
            if args.ds == True:
                ss = [[0],[1],[2],[3],[4]]
            else:
                ss = [[0]]

            #print(ss)
            loss = 0.0
            
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += F.interpolate(P[s[idx]], (y.shape[-3],y.shape[-2],y.shape[-1]), mode='trilinear')
                loss += loss_function(iout, y)
                
        scaler.scale(loss).backward()
        
        # Update weights with scaled optimizer step
        scaler.step(optimizer)
        
        # Update the scale for next iteration
        scaler.update()
        
        epoch_loss += loss.item()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        writer.add_scalar('Training Segmentation Loss', loss.data, global_step)
        global_step += 1
    return global_step, dice_val_best, global_step_best

max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 1 
dice_val_best = 0.0
global_step_best = 1
epoch_loss_values = []
metric_values = []

#args.mode = 'test'
#args.overlap = 0.5
#args.overlap_mode = 'gaussian'

if args.mode == 'train':
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()

epoch_iterator_val = tqdm(
    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
)

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        #jaccard = metric.binary.jc(pred, gt)
        #asd = metric.binary.assd(pred, gt)
        return dice, hd95#, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0#, 1, 0
    else:
        return 0, 0#, 0, 0

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0

def validation_last(epoch_iterator_val):
    model.eval()
    dice_vals = []
    hd_vals = []
    
    per_class_dice = []
    per_class_hd = []
    
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]

            if out_classes == 9:
                val_labels[val_labels == 5] = 0
                val_labels[val_labels == 9] = 0
                val_labels[val_labels == 10] = 0
                val_labels[val_labels == 12] = 0
                val_labels[val_labels == 13] = 0
                val_labels[val_labels == 11] = 5
            
            with autocast(enabled=False):
                val_outputs = sliding_window_inference_1out(
                    val_inputs, 
                    (args.img_size[0], args.img_size[1], args.img_size[2]), 
                    args.val_batch, 
                    model, 
                    overlap=args.overlap,
                    mode=args.overlap_mode
                )
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
                val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
                val_outputs = resample_3d(val_outputs, target_shape)
                dice_list_sub = []
                hd_list_sub = []
                for i in range(1, out_classes):
                    organ_Dice, organ_hd = calculate_metric_percase(val_outputs == i, val_labels == i) #dice(val_outputs == i, val_labels == i)
                    dice_list_sub.append(organ_Dice)
                    hd_list_sub.append(organ_hd)
                mean_dice = np.mean(dice_list_sub)
                mean_hd = np.mean(hd_list_sub)
                #nib.save(
                #    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
                #)

                # Store the class-wise metrics
                per_class_dice.append(dice_list_sub)
                #print(dice_per_class)
                per_class_hd.append(hd_list_sub)
                #print(hd_per_class)

                dice_vals.append(mean_dice)
                hd_vals.append(mean_hd)
                
                epoch_iterator_val.set_description(
                    "Validate (%d Steps) (mean_dice=%2.5f mean_hd=%2.5f)" % (step, np.mean(dice_vals), np.mean(hd_vals))
                )
        
    # Compute mean Dice and HD
    #print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))                
    mean_dice_val = np.mean(dice_vals)
    mean_hd_val = np.mean(hd_vals)

    # Return Dice per class, HD per class, mean Dice, and mean HD
    return np.array(per_class_dice), np.array(per_class_hd), mean_dice_val, mean_hd_val


import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from matplotlib import cm

def generate_colormap(num_classes):
    """
    Generate a colormap for a given number of classes using specified colors.
    Args:
        num_classes: Number of classes including the background.
    Returns:
        colormap: Dictionary mapping class indices to RGB values.
    """
    # List of preferred colors for classes
    color_list = [
        (0, 0, 0),          # Background (black)
        (255, 0, 0),        # Red
        (0, 255, 0),        # Green
        (135, 206, 250),    # Skyblue
        (255, 165, 0),      # Orange
        (255, 0, 255),      # Magenta
        (128, 0, 128),      # Purple
        (255, 255, 0),      # Yellow
        (205, 133, 63),     # Peru
        (128, 128, 0),      # Olive
        (75, 0, 130),       # Indigo
        (0, 255, 128),      # Lime
        (0, 0, 255),        # Blue
        (255, 20, 147)      # Deeppink
    ]
    
    # Ensure we have enough colors for the number of classes
    assert num_classes <= len(color_list), "Not enough colors specified for the number of classes."
    
    # Map each class index to its corresponding RGB value
    colormap = {cls: color_list[cls] for cls in range(num_classes)}
    return colormap


def overlay_segmentation(image, segmentation, colormap):
    """
    Overlay segmentation on the input image using a specified colormap.
    Args:
        image: Original 8-bit image (2D numpy array).
        segmentation: Segmentation mask (2D numpy array with class indices).
        colormap: Dictionary mapping class indices to RGB values.
    Returns:
        overlay: RGB image with segmentation overlay.
    """
    overlay = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
    for cls, color in colormap.items():
        if cls == 0:  # Skip background (keep original image pixels)
            continue
        overlay[segmentation == cls] = np.array(color)
    return overlay

def validation_save(epoch_iterator_val): 
    model.eval()
    dice_vals = []
    hd_vals = []
    per_class_dice = []
    per_class_hd = []

    # Dynamically generate colormap for 'out_classes'
    colormap = generate_colormap(out_classes)

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            volume_name = os.path.splitext(img_name)[0]
            output_folder = os.path.join(root_dir, os.path.join("output_slices", volume_name))
            os.makedirs(output_folder, exist_ok=True)

            if out_classes == 9:
                val_labels[val_labels == 5] = 0
                val_labels[val_labels == 9] = 0
                val_labels[val_labels == 10] = 0
                val_labels[val_labels == 12] = 0
                val_labels[val_labels == 13] = 0
                val_labels[val_labels == 11] = 5
            
            with autocast(enabled=False):
                val_outputs = sliding_window_inference_1out(
                    val_inputs, 
                    (args.img_size[0], args.img_size[1], args.img_size[2]), 
                    args.val_batch, 
                    model, 
                    overlap=args.overlap,
                    mode=args.overlap_mode
                )
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
                val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
                val_outputs = resample_3d(val_outputs, target_shape)
                
                for slice_idx in range(val_outputs.shape[2]):
                    # Get 2D slices
                    input_slice = val_inputs[0, 0, :, :, slice_idx].cpu().numpy()

                    # Normalize input slice to 8-bit
                    #input_slice_normalized = rescale_intensity(input_slice, in_range='image', out_range=(0, 255))
                    #input_slice_8bit = img_as_ubyte(input_slice_normalized)
                    # Normalize the input slice to the range [0, 1] before converting to 8-bit
                    input_slice_normalized = rescale_intensity(input_slice, in_range='image', out_range=(0, 1))
                    input_slice_8bit = img_as_ubyte(input_slice_normalized)
                    
                    # Overlay predictions and ground truth on the 8-bit input slice
                    pred_slice = val_outputs[:, :, slice_idx]
                    gt_slice = val_labels[:, :, slice_idx]
                    pred_overlay = overlay_segmentation(input_slice_8bit, pred_slice, colormap)
                    gt_overlay = overlay_segmentation(input_slice_8bit, gt_slice, colormap)

                    # Save images
                    input_output_path = os.path.join(output_folder, "input")
                    pred_output_path = os.path.join(output_folder, "output")
                    gt_output_path = os.path.join(output_folder, "gt")
                    
                    os.makedirs(input_output_path, exist_ok=True)
                    os.makedirs(pred_output_path, exist_ok=True)
                    os.makedirs(gt_output_path, exist_ok=True)

                    input_output_path = os.path.join(input_output_path, f"slice_{slice_idx:03d}.png")
                    pred_output_path = os.path.join(pred_output_path, f"slice_{slice_idx:03d}.png")
                    gt_output_path = os.path.join(gt_output_path, f"slice_{slice_idx:03d}.png")

                    # Save input image
                    plt.imsave(input_output_path, input_slice_8bit, cmap="gray")
                    # Save overlays
                    plt.imsave(pred_output_path, pred_overlay.astype(np.uint8))
                    plt.imsave(gt_output_path, gt_overlay.astype(np.uint8))

                dice_list_sub = []
                hd_list_sub = []
                for i in range(1, out_classes):  # Iterate over foreground classes
                    organ_Dice, organ_hd = calculate_metric_percase(val_outputs == i, val_labels == i)
                    dice_list_sub.append(organ_Dice)
                    hd_list_sub.append(organ_hd)
                mean_dice = np.mean(dice_list_sub)
                mean_hd = np.mean(hd_list_sub)

                # Store the class-wise metrics
                per_class_dice.append(dice_list_sub)
                per_class_hd.append(hd_list_sub)
                dice_vals.append(mean_dice)
                hd_vals.append(mean_hd)
                
                epoch_iterator_val.set_description(
                    "Validate (%d Steps) (mean_dice=%2.5f mean_hd=%2.5f)" % (step, np.mean(dice_vals), np.mean(hd_vals))
                )
        
    mean_dice_val = np.mean(dice_vals)
    mean_hd_val = np.mean(hd_vals)

    return np.array(per_class_dice), np.array(per_class_hd), mean_dice_val, mean_hd_val

# Define class labels dictionary

def save_metrics_to_csv(trained_weights, dataset_name, network_name, overlap, overlap_mode, class_labels, per_class_dice, per_class_hd, mean_dice_val, mean_hd_val, csv_filename):
    # Check if the file exists, if not, write the header
    file_exists = os.path.isfile(csv_filename)
    
    # Header for the CSV file
    header = ["Trained_Weights", "Dataset", "Network", "Overlap", "OverlapdMode"]
    
    # Add Dice class-wise and mean columns using class labels
    header += [f"Dice_{class_labels[str(i)]}" for i in range(len(class_labels))] + ["Mean_Dice"]
    
    # Add HD class-wise and mean columns using class labels
    header += [f"HD_{class_labels[str(i)]}" for i in range(len(class_labels))] + ["Mean_HD"]
    
    # Prepare the row data
    row = [
        trained_weights,  # Trained weights file name
        dataset_name,     # Dataset name
        network_name,      # Network name
        overlap,
        overlap_mode
    ]
    
    # Append class-wise Dice scores
    row += [f"{dice:.4f}" for dice in per_class_dice.mean(axis=0)]  # average Dice per class over the validation steps
    
    # Append mean Dice score
    row.append(f"{mean_dice_val:.4f}")
    
    # Append class-wise HD scores
    row += [f"{hd:.4f}" for hd in per_class_hd.mean(axis=0)]  # average HD per class over the validation steps
    
    # Append mean HD score
    row.append(f"{mean_hd_val:.4f}")
    
    # Write to CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(header)  # Write the header if file does not exist
        
        writer.writerow(row)  # Write the data row

# Example usage:
csv_filename = "last_validation_metrics_btcv.csv"

# Suppose these are returned from the validation function
per_class_dice, per_class_hd, mean_dice_val, mean_hd_val = validation_save(epoch_iterator_val)


if out_classes == 9:
    class_labels = {
        "0": "background",
        "1": "spleen",
        "2": "rkid",   # Right kidney
        "3": "lkid",   # Left kidney
        "4": "gall",   # Gallbladder
        "5": "pancreas",
        "6": "liver",
        "7": "sto",    # Stomach
        "8": "aorta",
    }
else:
    class_labels = {
        "0": "BG",
        "1": "Spl",
        "2": "Rkid",   # Right kidney
        "3": "Lkid",   # Left kidney
        "4": "Gall",   # Gallbladder
        "5": "Eso",    # Esophagus
        "6": "Liver",
        "7": "Sto",    # Stomach
        "8": "Aorta",
        "9": "IVC",    # Inferior vena cava
        "10": "Veins",
        "11": "Pan",
        "12": "Rad",   # Right adrenal gland
        "13": "Lad"    # Left adrenal gland
    }

# Save the metrics into the CSV file
save_metrics_to_csv(args.output, args.dataset, args.network, args.overlap, args.overlap_mode, class_labels, per_class_dice, per_class_hd, mean_dice_val, mean_hd_val, csv_filename)
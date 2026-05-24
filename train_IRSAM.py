# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as ops
from tqdm import tqdm
import argparse
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
import math
from typing import Dict, List, Tuple

from segment_anything_training.build_IRSAM import build_sam_IRSAM

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric
from utils.metric import PD_FA, ROCMetric
from utils.loss_mask import DICE_loss, sigmoid_ce_loss_jit, dice_loss_jit
from utils.log import initialize_logger
import utils.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('IRSAM Training', add_help=False)

    parser.add_argument("--output", type=str, default="output",
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--checkpoint", type=str, default="mobile_sam.pt",
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")

    # Training hyperparameters (from the IRSAM paper)
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Initial learning rate (paper: 1e-4)')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='AdamW weight decay')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--max_epoch_num', default=500, type=int,
                        help='Total training epochs (paper: 500)')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='Number of warmup epochs for LR scheduler')
    parser.add_argument('--dataloader_size', default=[512, 512], type=list,
                        help='Input image size (paper: 512x512)')
    parser.add_argument('--batch_size_train', default=4, type=int,
                        help='Training batch size (paper: 4)')
    parser.add_argument('--batch_size_valid', default=4, type=int,
                        help='Validation batch size')
    parser.add_argument('--model_save_fre', default=5, type=int,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_fre', default=1, type=int,
                        help='Evaluate every N epochs')

    # Loss weights
    parser.add_argument('--dice_weight', default=1.0, type=float,
                        help='Weight for mask dice loss')
    parser.add_argument('--bce_weight', default=1.0, type=float,
                        help='Weight for mask BCE loss')
    parser.add_argument('--edge_weight', default=1.0, type=float,
                        help='Weight for edge BCE loss')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='NUDT',
                        choices=['NUDT', 'IRSTD', 'Sirstv2', 'all'],
                        help='Which dataset to train on')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    # Eval only
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    return parser.parse_args()


# --------------- Dataset Configurations ---------------

def get_train_datasets(dataset_name):
    """Return (train_datasets, valid_datasets) based on dataset_name."""

    dataset_train_NUDT = {"name": "NUDT",
                          "im_dir": "datasets/NUDT-SIRST00/trainval_images",
                          "gt_dir": "datasets/NUDT-SIRST00/trainval_masks",
                          "im_ext": ".png",
                          "gt_ext": ".png"}

    dataset_val_NUDT = {"name": "NUDT",
                        "im_dir": "datasets/NUDT-SIRST00/test_images",
                        "gt_dir": "datasets/NUDT-SIRST00/test_masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_train_IRSTD = {"name": "IRSTD",
                           "im_dir": "datasets/IRSTD-1k/trainval_images",
                           "gt_dir": "datasets/IRSTD-1k/trainval_masks",
                           "im_ext": ".png",
                           "gt_ext": ".png"}

    dataset_val_IRSTD = {"name": "IRSTD",
                         "im_dir": "datasets/IRSTD-1k/test_images",
                         "gt_dir": "datasets/IRSTD-1k/test_masks",
                         "im_ext": ".png",
                         "gt_ext": ".png"}

    dataset_train_Sirstv2 = {"name": "Sirstv2_512",
                             "im_dir": "datasets/Sirstv2_512/trainval_images",
                             "gt_dir": "datasets/Sirstv2_512/trainval_masks",
                             "im_ext": ".png",
                             "gt_ext": ".png"}

    dataset_val_Sirstv2 = {"name": "Sirstv2_512",
                           "im_dir": "datasets/Sirstv2_512/test_images",
                           "gt_dir": "datasets/Sirstv2_512/test_masks",
                           "im_ext": ".png",
                           "gt_ext": ".png"}

    if dataset_name == 'NUDT':
        return [dataset_train_NUDT], [dataset_val_NUDT]
    elif dataset_name == 'IRSTD':
        return [dataset_train_IRSTD], [dataset_val_IRSTD]
    elif dataset_name == 'Sirstv2':
        return [dataset_train_Sirstv2], [dataset_val_Sirstv2]
    elif dataset_name == 'all':
        return ([dataset_train_NUDT, dataset_train_IRSTD, dataset_train_Sirstv2],
                [dataset_val_NUDT, dataset_val_IRSTD, dataset_val_Sirstv2])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# --------------- Evaluation ---------------

def evaluate(net, valid_dataloaders):
    """Evaluate model on validation set(s). Returns dict of metrics."""
    net.eval()
    metric = dict()

    IoU_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    ROC = ROCMetric(1, 10)
    Pd_Fa = PD_FA(1, 10)

    IoU_metric.reset()
    nIoU_metric.reset()
    Pd_Fa.reset()

    for k in range(len(valid_dataloaders)):
        valid_dataloader = valid_dataloaders[k]

        dataset_IoU_metric = SigmoidMetric()
        dataset_nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        dataset_Pd_Fa = PD_FA(1, 10)

        dataset_IoU_metric.reset()
        dataset_nIoU_metric.reset()
        dataset_Pd_Fa.reset()

        tbar = tqdm(valid_dataloader, desc='Evaluating')
        for data_val in tbar:
            imidx_val = data_val['imidx']
            inputs_val = data_val['image']
            labels_val = data_val['label']
            shapes_val = data_val['shape']
            labels_ori = data_val['ori_label']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = (torch.as_tensor((imgs[b_i]).astype(dtype=np.uint8), device=net.device)
                               .permute(2, 0, 1).contiguous())
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                masks, edges = net(batched_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Resize predicted masks/edges to match original labels resolution for accurate evaluation
            if masks.shape[-2:] != labels_ori.shape[-2:]:
                masks = F.interpolate(masks, size=labels_ori.shape[-2:], mode='bilinear', align_corners=False)
                edges = F.interpolate(edges, size=labels_ori.shape[-2:], mode='bilinear', align_corners=False)

            # Update overall metrics
            IoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            nIoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            Pd_Fa.update(masks.cpu(), (labels_ori / 255.).cpu().detach())

            # Update dataset-specific metrics
            dataset_IoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            dataset_nIoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            dataset_Pd_Fa.update(masks.cpu(), (labels_ori / 255.).cpu().detach())

            d_FA, d_PD = dataset_Pd_Fa.get(len(valid_dataloader))
            _, d_IoU = dataset_IoU_metric.get()
            _, d_nIoU = dataset_nIoU_metric.get()

            tbar.set_description('IoU:%f, nIoU:%f, PD:%.8lf, FA:%.8lf'
                                 % (d_IoU, d_nIoU, d_PD[0], d_FA[0]))

        # Get dataset name
        try:
            dataset_name = valid_dataloader.dataset.dataset["data_name"][0]
        except Exception:
            dataset_name = f"dataset_{k}"

        # Print and log the dataset-specific metrics
        log_msg = f"Dataset {dataset_name} Eval: IoU={d_IoU:.4f}, nIoU={d_nIoU:.4f}, PD={d_PD[0]:.8f}, FA={d_FA[0]:.8f}"
        print(log_msg)
        logger = logging.getLogger()
        logger.info(log_msg)

        # Store in metric dict
        metric[f"{dataset_name}_iou"] = d_IoU
        metric[f"{dataset_name}_niou"] = d_nIoU
        metric[f"{dataset_name}_pd"] = d_PD[0]
        metric[f"{dataset_name}_fa"] = d_FA[0]

        # Calculate final overall accumulated metrics to populate return dict
        FA, PD = Pd_Fa.get(len(valid_dataloader))
        _, IoU = IoU_metric.get()
        _, nIoU = nIoU_metric.get()

        metric['iou'] = IoU
        metric['niou'] = nIoU
        metric['pd'] = PD[0]
        metric['fa'] = FA[0]

    return metric


# --------------- Training ---------------

def train_one_epoch(net, train_dataloader, optimizer, epoch, args, logger=None):
    """Train for one epoch. Returns average loss."""
    net.train()

    total_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    total_edge_loss = 0.0
    num_batches = 0

    tbar = tqdm(train_dataloader, desc=f'Epoch {epoch}')
    for data in tbar:
        inputs = data['image']
        labels = data['label']
        edges_gt = data['edge']

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            edges_gt = edges_gt.cuda()

        # Normalize labels to [0, 1]
        labels = labels / 255.0
        edges_gt = edges_gt / 255.0

        # Construct batched_input for SAM-style forward pass
        imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
        batched_input = []
        for b_i in range(len(imgs)):
            dict_input = dict()
            input_image = (torch.as_tensor((imgs[b_i]).astype(dtype=np.uint8), device=net.device)
                           .permute(2, 0, 1).contiguous())
            dict_input['image'] = input_image
            dict_input['original_size'] = imgs[b_i].shape[:2]
            batched_input.append(dict_input)

        # Forward pass
        masks_pred, edges_pred = net(batched_input)

        # --- Compute losses ---

        # Mask loss: Dice + BCE
        # DICE_loss expects raw logits (applies sigmoid internally)
        loss_dice, loss_iou = DICE_loss(masks_pred, labels)

        # BCE loss for masks
        loss_bce_mask = F.binary_cross_entropy_with_logits(
            masks_pred, labels, reduction='mean'
        )

        # Edge loss: BCE (edges_pred already has sigmoid applied in decoder)
        # So use regular BCE, not with_logits
        loss_edge = F.binary_cross_entropy(
            edges_pred.clamp(1e-6, 1 - 1e-6), edges_gt, reduction='mean'
        )

        # Total loss
        loss = (args.dice_weight * loss_dice +
                args.bce_weight * loss_bce_mask +
                args.edge_weight * loss_edge)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_dice_loss += loss_dice.item()
        total_bce_loss += loss_bce_mask.item()
        total_edge_loss += loss_edge.item()
        num_batches += 1

        tbar.set_postfix(
            loss=f'{loss.item():.4f}',
            dice=f'{loss_dice.item():.4f}',
            bce=f'{loss_bce_mask.item():.4f}',
            edge=f'{loss_edge.item():.4f}'
        )

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice_loss / max(num_batches, 1)
    avg_bce = total_bce_loss / max(num_batches, 1)
    avg_edge = total_edge_loss / max(num_batches, 1)

    log_msg = (f'Epoch [{epoch}/{args.max_epoch_num}] '
               f'Loss: {avg_loss:.4f} (dice: {avg_dice:.4f}, '
               f'bce: {avg_bce:.4f}, edge: {avg_edge:.4f})')
    print(log_msg)
    if logger:
        logger.info(log_msg)

    return avg_loss


def build_optimizer(net, args):
    """
    Build AdamW optimizer with per-parameter learning rate scaling.
    The TinyViT encoder assigns lr_scale to parameters via set_layer_lr_decay(),
    allowing layer-wise learning rate decay.
    """
    param_groups = []
    no_decay_keywords = {'bias', 'norm', 'LayerNorm', 'BatchNorm', 'bn'}

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        # Determine learning rate scale
        lr_scale = getattr(param, 'lr_scale', 1.0)

        # Determine weight decay (no decay for biases and norms)
        apply_decay = True
        for kw in no_decay_keywords:
            if kw in name:
                apply_decay = False
                break

        param_groups.append({
            'params': [param],
            'lr': args.learning_rate * lr_scale,
            'weight_decay': args.weight_decay if apply_decay else 0.0,
            'param_name': name,
        })

    optimizer = optim.AdamW(param_groups)
    return optimizer


def build_scheduler(optimizer, args):
    """
    Build cosine annealing LR scheduler with optional warmup.
    """
    # Cosine annealing from current LR to 0 over remaining epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epoch_num - args.warmup_epochs,
        eta_min=1e-7,
    )
    return scheduler


def warmup_lr(optimizer, epoch, warmup_epochs, base_lrs):
    """Linear warmup of learning rate."""
    if epoch >= warmup_epochs:
        return
    alpha = epoch / warmup_epochs
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = base_lrs[i] * alpha


def rename_output_dir(old_dir, suffix):
    if not suffix:
        return old_dir
    
    logger = logging.getLogger()
    # Close and remove all handlers to release file lock on info_*.log
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
        
    new_dir = old_dir.rstrip('/') + suffix
    try:
        # Check if the new directory already exists, if so append a number to avoid collision
        if os.path.exists(new_dir):
            base_new_dir = new_dir
            counter = 1
            while os.path.exists(f"{base_new_dir}_{counter}"):
                counter += 1
            new_dir = f"{base_new_dir}_{counter}"
            
        os.rename(old_dir, new_dir)
        print(f"Successfully renamed output directory to: {new_dir}")
        return new_dir
    except Exception as e:
        print(f"Warning: could not rename output directory to {new_dir}: {e}")
        return old_dir


def main(args):
    # --- Setup output directory ---
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Format a concise but informative directory name
    run_name = f"run_{timestamp}_{args.dataset}_lr{args.learning_rate}_wd{args.weight_decay}_bs{args.batch_size_train}_epochs{args.max_epoch_num}"
    run_name = run_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    base_output = args.output
    args.output = os.path.join(base_output, run_name)
    
    os.makedirs(args.output, exist_ok=True)
    initialize_logger(args.output)
    logger = logging.getLogger()

    logger.info(f"Training IRSAM with args: {args}")

    # Save passed arguments to args.txt
    args_file = os.path.join(args.output, "args.txt")
    try:
        with open(args_file, "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")
        logger.info(f"Saved training arguments to {args_file}")
    except Exception as e:
        logger.warning(f"Could not save arguments to file: {e}")

    # --- Step 1: Build datasets ---
    train_datasets, valid_datasets = get_train_datasets(args.dataset)

    print("--- Creating training dataloader ---")
    train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    train_dataloaders, train_datasets_obj = create_dataloaders(
        train_im_gt_list,
        my_transforms=[
            Resize(args.dataloader_size),
            RandomHFlip(prob=0.5),
        ],
        batch_size=args.batch_size_train,
        training=True
    )
    print(f"Training dataloader created with {len(train_datasets_obj)} samples")

    print("--- Creating validation dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets_obj = create_dataloaders(
        valid_im_gt_list,
        my_transforms=[
            Resize(args.dataloader_size),
        ],
        batch_size=args.batch_size_valid,
        training=False
    )
    print(f"{len(valid_dataloaders)} validation dataloader(s) created")

    # --- Step 2: Build model ---
    print("--- Building IRSAM model ---")
    net = build_sam_IRSAM(checkpoint=args.checkpoint)

    # Move to device
    if torch.cuda.is_available():
        net.cuda()

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # --- Step 3: Build optimizer and scheduler ---
    optimizer = build_optimizer(net, args)
    scheduler = build_scheduler(optimizer, args)

    # Store base LRs for warmup
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    best_iou = 0.0
    best_metrics = None
    start_epoch = args.start_epoch

    # --- Resume from checkpoint if specified ---
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_iou = ckpt.get('best_iou', 0.0)
        best_metrics = ckpt.get('metric', None)
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")
        logger.info(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")

    # --- Eval only mode ---
    if args.eval:
        print("--- Evaluation only ---")
        metric = evaluate(net, valid_dataloaders)
        print(f"Results: IoU={metric['iou']:.4f}, nIoU={metric['niou']:.4f}, "
              f"PD={metric['pd']:.8f}, FA={metric['fa']:.8f}")
        
        # Rename directory with evaluation scores
        scores_suffix = f"_IoU{metric['iou']:.4f}_nIoU{metric['niou']:.4f}"
        rename_output_dir(args.output, scores_suffix)
        return

    # --- Step 4: Training loop ---
    print("=" * 60)
    print(f"Starting training for {args.max_epoch_num} epochs")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size_train}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Optimizer: AdamW (weight_decay={args.weight_decay})")
    print(f"LR schedule: Cosine Annealing (warmup={args.warmup_epochs})")
    print("=" * 60)

    for epoch in range(start_epoch, args.max_epoch_num):
        # Warmup learning rate
        if epoch < args.warmup_epochs:
            warmup_lr(optimizer, epoch, args.warmup_epochs, base_lrs)

        # Train one epoch
        net.train()
        avg_loss = train_one_epoch(net, train_dataloaders, optimizer, epoch, args, logger)

        # Step LR scheduler (after warmup)
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch} LR: {current_lr:.8f}')

        # --- Periodic evaluation ---
        if (epoch + 1) % args.eval_fre == 0 or epoch == args.max_epoch_num - 1:
            print(f"\n--- Evaluation at epoch {epoch} ---")
            metric = evaluate(net, valid_dataloaders)
            logger.info(f'Epoch {epoch} Eval: IoU={metric["iou"]:.4f}, '
                        f'nIoU={metric["niou"]:.4f}, '
                        f'PD={metric["pd"]:.8f}, FA={metric["fa"]:.8f}')
            print(f'IoU={metric["iou"]:.4f}, nIoU={metric["niou"]:.4f}, '
                  f'PD={metric["pd"]:.8f}, FA={metric["fa"]:.8f}')

            # Save best model
            if metric['iou'] > best_iou:
                best_iou = metric['iou']
                best_metrics = metric
                save_path = os.path.join(args.output, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_iou': best_iou,
                    'metric': metric,
                }, save_path)
                print(f'New best model saved! IoU: {best_iou:.4f}')
                logger.info(f'New best model saved at epoch {epoch}! IoU: {best_iou:.4f}')

            # Set back to training mode
            net.train()

        # --- Periodic checkpoint saving ---
        if (epoch + 1) % args.model_save_fre == 0:
            save_path = os.path.join(args.output, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
            }, save_path)
            print(f'Checkpoint saved: {save_path}')

    # --- Save final model ---
    save_path = os.path.join(args.output, 'final_model.pth')
    torch.save({
        'epoch': args.max_epoch_num - 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou,
    }, save_path)
    print(f'\nTraining complete! Final model saved: {save_path}')
    print(f'Best IoU: {best_iou:.4f}')
    logger.info(f'Training complete! Best IoU: {best_iou:.4f}')

    # Rename directory with evaluation scores
    if best_metrics is not None:
        scores_suffix = f"_IoU{best_metrics['iou']:.4f}_nIoU{best_metrics['niou']:.4f}"
    elif best_iou > 0.0:
        scores_suffix = f"_IoU{best_iou:.4f}"
    else:
        scores_suffix = ""
    rename_output_dir(args.output, scores_suffix)


if __name__ == "__main__":
    args = get_args_parser()
    main(args)

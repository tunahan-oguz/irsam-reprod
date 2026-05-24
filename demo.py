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
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms  as T
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from thop import profile

from segment_anything_training.build_IRSAM import build_sam_IRSAM

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric
from utils.metric import PD_FA, ROCMetric
from utils.loss_mask import DICE_loss
from utils.log import initialize_logger
import utils.misc as misc
from train_IRSAM import evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument('--dataloader_size', default=[512, 512], type=list)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--restore_model', type=str, default=None)
    return parser.parse_args()


def main(valid_datasets, args):
    # --- Step 1: Valid dataset ---
    print("--- create valid dataloader ---")
    print("CUDA available:", torch.cuda.is_available())
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[
                                                               Resize(args.dataloader_size)
                                                           ],
                                                           batch_size=args.batch_size_valid,
                                                           training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # --- Step 2: Load pretrained Network---
    net = build_sam_IRSAM(checkpoint=args.checkpoint)
    net.cuda()

    metric = evaluate(net, valid_dataloaders)
    print(f"Results: IoU={metric['iou']:.4f}, nIoU={metric['niou']:.4f}, "
            f"PD={metric['pd']:.8f}, FA={metric['fa']:.8f}")



if __name__ == "__main__":
    # --------------- Configuring the Valid datasets ---------------
    dataset_val_nuaa = {"name": "Sirstv2_512",
                        "im_dir": "datasets/Sirstv2_512/test_images",
                        "gt_dir": "datasets/Sirstv2_512/test_masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_val_NUDT = {"name": "NUDT",
                        "im_dir": "datasets/NUDT-SIRST00/test_images",
                        "gt_dir": "datasets/NUDT-SIRST00/test_masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_val_IRSTD = {"name": "IRSTD",
                         "im_dir": "datasets/IRSTD-1k/test_images",
                         "gt_dir": "datasets/IRSTD-1k/test_masks",
                         "im_ext": ".png",
                         "gt_ext": ".png"}

    valid_datasets = [dataset_val_IRSTD]

    args = get_args_parser()

    main(valid_datasets, args)

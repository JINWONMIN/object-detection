import utils
import dataset.transforms as T
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from dataset.coco_utils import get_coco, get_coco_kp
from engine import train_one_epoch, evaluate
from dataset.group_by_aspect_ratio import GroupedBatchSampler, compute_aspect_ratios
from config.train_option import parse_args
from config.config import *
import torchvision

import cv2
import random


def get_dataset(name, image_set, transform):
    paths = {
        "coco": ('/public/yzy/coco/2017', get_coco, 91),
        "coco_kp": ('/datasets01/COCO/022719', get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    args = parse_args()
    if args.output_dir:
        utils.mkdir(args.output_dir)
    utils.init_distributed_mode(args)

    # Data Loading
    print("Loading data")
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True))
    

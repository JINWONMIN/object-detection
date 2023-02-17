"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
import numpy as np
import argparse
import torch
from src.dataset import CocoDataset
from src.transform import SSDTransformer
import cv2
import shutil

from src.utils import generate_dboxes, Encoder, colors
from src.model import SSD, ResNet


def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--data-path", type=str, default="/coco", 
                        help="the root folder of dataset")
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--output", type=str, default="predictions")
    args = parser.parse_args()
    return args


def test(opt):
    pass
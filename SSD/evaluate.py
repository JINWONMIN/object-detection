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
    pass
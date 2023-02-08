import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Loss(nn.Module):
    """ Constructor

    Args:
        feature_size (int): size of input feature map.
        num_bboxes (int): number of bboxes per each cell.
        num_classes (int): number of the object classes.
        lambda_coord (float): weight for bbox location/size losses.
        lambda_noobj (float): weight for no-objectness loss.
    """
    def __init__(self, feature_size=7, num_bboxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        super(Loss, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].

        Args:
            bbox1 (Tensor): bounding bboxes, sized [N, 4].
            bbox2 (Tensor): bounding bboxes, sized [N, 4].
        """

        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)   # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)   # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersection from the coordinates
        wh = rb - lt    # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]   # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])   # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])   # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter   # [N, M, 2]    
        iou = inter / union             # [N, M, 2]

        return iou
    
    def forward(self, pred_tensor, target_tensor):
        """ Compute loss for YOLO training.

        Args:
            pred_tensor (_type_): _description_
            target_tensor (_type_): _description_
        """
        pass
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import cv2
import numpy as np

from backbone.darknet import DarkNet
from yolo_v1 import YOLOv1


class YOLODetector:
    def __init__(self,
                 model_path=None, class_name_list=None, mean_rgb=[122.67891434, 116.66876762, 104.00698793],
                 conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.5,
                 gpu_id=0):
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        use_gpu = torch.cuda.is_available()
        assert use_gpu, 'Current implementation does not support CPU mode. Enable CUDA'
        
        # Load YOLO model
        print("Loading YOLO model...")
        darknet = DarkNet(conv_only=True, bn=True, init_weight=True) 
        darknet.features = torch.nn.DataParallel(darknet.features)  # parallel processing
        self.yolo = YOLOv1(darknet.features)    # darknet feature를 yolo에 push
        self.yolo.conv_layers = torch.nn.DataParallel(self.yolo.conv_layers)

        if os.path.exists(model_path):
            self.yolo.load_state_dict(torch.load(model_path))

        self.yolo.cuda()
        print("Done loading!")
        
        self.yolo.eval()
        
        self.S = self.yolo.feature_size # 7x7
        self.B = self.yolo_num_bboxes   # 2
        self.C = self.yolo_num_classes  # 20
        
        self.class_name_list = class_name_list if (class_name_list is not None) else list(VOC_CLASS_BGR.keys())
        assert len(self.class_name_list) == self.C
        
        self.mean = np.array(mean_rgb, dtype=np.float32)
        assert self.mean.shape == (3,)

        self.conf_thresh = conf_thresh
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        
        self.to_tensor = transforms.ToTensor()
        
        # Warm up for 10 epochs
        # parameter가 Random values로 설정되므로, 초기부터 큰 lr는 학습의 불안정을 초래함.
        # Warm-up을 시키면 초기 학습이 안정적으로 수행됨.
        dummy_input = Variable(torch.zeros((1, 3, 448, 448)))   # input shape
        dummy_input = dummy_input.cuda()
        for i in range(10):
            self.yolo(dummy_input)
            
    
    def detect(self, image_bgr, image_size=448):
        """Detect objects from given image.

        Args:
            image_bgr (numpy array): input image in BGR ids_sorted, sized [h, w, 3].
            image_size (int, optional): image width and height to which input image is resized. Defaults to 448.
        Returns:
            boxes_detected (list of tuple): box cornor list like [((x1, y1), (x2, y2))_obj1, ...]. Re-scaled for original input image size.
            class_names_detected (list of str): list of class name for each detected box.
            probs_detected (list of float): list of probability(=confidence x class_score) for each detected box.
        """
        h, w, _ = image_bgr.shape
        img = cv2.resize(image_bgr, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # assuming the model is trained with RGB images.
        img = (img - self.mean) / 255.0     # image normalization
        img = self.to_tensor(img)   # [image_size, image_size, 3] -> [3, image_size, image_size] ex) (3, 448, 448)
        img = img[None, :, :, :]    # [3, image_size, image_size] -> [1, 3, image_size, image_size] ex) (1, 3, 448, 448)
        img = Variable(img)
        img = img.cuda()
        
        with torch.no_grad():
            pred_tensor = self.yolo(img)
        pred_tensor = pred_tensor.cpu().data
        pred_tensor = pred_tensor.squeeze(0)    # squeeze batch dimension: batch 차원 제거
        
        # Get detected boxes_detected, labels, confidences, class-scores.
        boxes_normalized_all, class_labels_all, confidences_all, class_scores_all = self.decode(pred_tensor)
        if boxes_normalized_all.size(0) == 0:
            return [], [], []   # if no box found, return empty lists.
        
        # Apply non maximum supression for boxes of each class.
        boxes_normalized, class_labels, probs = [], [], []
        
        for class_label in range(len(self.class_name_list)):
            mask = (class_labels_all == class_label)
            if torch.sum(mask) == 0:
                continue    # if no box found, skip that class
        
            
        
    def decode(self, pred_tensor):
        """ Decode tensor into box coordinates, class labels, and probs_detected.

        Args:
            pred_tensor (tensor): tensor to decode sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        Returns:
            boxes (tensor): [[x1, y1, x2, y2]_obj1, ...]. Normalized from 0.0 to 1.0 w.r.t image width/height, sized [n_boxes, 4].
            labels (tensor): class labels for each detected box, sized [n_boxes].
            confidences (tensor): objectness confidences for each detected box, sized [n_boxes].
            class_scores (tensor): scores for most likely class for each detected box, sized [n_boxes].
        """
        S, B, C = self.S, self.B, self.C
        boxes, labels, confidences, class_scores = [], [], [], []
        
        cell_size = 1.0 / float(S)
        
        # torch.unsqueeze() : Add a 1D to a specific location
        conf = pred_tensor[:, :, 4].unsqueeze(2)    # [S, S, 1]
        for b in range(1, B):
            conf = torch.cat((conf, pred_tensor[:, :, 5*b + 4].unsqueeze(2)), 2)
        conf_mask = conf > self.conf_thresh     # [S, S, B]

        # TBM, further optimization may be possible by replacing the following for-loops with tensor operations.
        for i in range(S):  # for x-dimension
            for j in range(S):  # for y-dimension
                class_score, class_label = torch.max(pred_tensor[j, i, 5*B:], 0)

                for b in range(B):
                    conf = pred_tensor[j, i, 5*b + 4]
                    prob = conf * class_score
                    if float(prob) < self.prob_thresh:
                        continue
                    
                    # Compute box corner (x1, y1, x2, y2) from tensor
                    box = pred_tensor[j, i, 5*b : 5*b + 4]  # [x1, y1, x2, y2] for detector each grid cell
                    x0y0_normalized = torch.FloatTensor([i, j]) * cell_size # 

    def nms(self, boxes, scores):
        """Apply non maximum supression.

        Args:
            boxes (_type_): _description_
            scores (_type_): _description_
        """
        threshold = self.nms_thresh
        
        x1 = boxes[:, 0]    # [n,]
        y1 = boxes[:, 1]    # [n,]
        x2 = boxes[:, 2]    # [n,]
        y2 = boxes[:, 3]    # [n,]
        areas = (x2 -x1) * (y2 - y1)    # [n,]

        _, ids_sorted = scores.sort(0, descending=True) # [n,]
        ids = []
        # torch.numel() -> returns: It returns the length of the input tensor.
        while ids_sorted.numel() > 0:
            # Assume 'ids_sorted' size is [m,] in the beginning of this iter.

            i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
            ids.append(i)

            if ids_sorted.numel() == 1:
                break   # If only one box is left (i.e., no box to supress), break.
            
            # torch.clamp(input, min, max, *, out=None) -> Tensor
            # Clamp all elements in input into the range [min, max] and return a resulting tensor.
            inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i])  # [m-1, ]
            inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i])  # [m-1, ]
            inter_x2 = x2[ids_sorted[1:]].clamp(max=y2[i])  # [m-1, ]
            inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i])  # [m-1, ]
            inter_w = (inter_x2 - inter_x1).clamp(min=0)    # [m-1, ]
            inter_h = (inter_y2 - inter_y1).clamp(min=0)    # [m-1, ]

            inters = inter_w * inter_h  # intersection b/w/ box 'i' and other boxes, sized [m-1, ].
            unions = areas[i] + areas[ids_sorted[1:]] - inters  # unions b/w/ box 'i' and other boxes, sized [m-1, ]
            ious = inters / unions  # [m-1, ]

            # Remove boxes whose IoU is higher than the threshold.
            ids_keep = (ious <= threshold).nonzero().squeeze()  # [m-1, ]. Because 'nonzero()' adds extra dimension, squeeze it.
            if ids_keep.numel() == 0:
                break   # If no box left, break.
            ids_sorted = ids_sorted[ids_keep + 1]   # '+1' is needed because 'ids_sorted[0] = i'.
        
        return torch.LongTensor(ids)
            
# VOC class names and BGR color.
VOC_CLASS_BGR = {
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (192, 128, 128),
    'pottedplant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'tvmonitor': (0, 64, 128)
}

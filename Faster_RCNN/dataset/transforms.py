import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    """
    인자로 받은 딕셔너리의 밸류를 변형해주는 클래스

    Args:
        transforms (dict): image, coord 
    """
    def __init__(self, transforms) -> None:
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
    
class RandomHorizontalFlip(object):
    """
    이자로 받은 객체의 바운딩 박스와 이미지를 좌우 반전 시켜주는 클래스

    Args:
        prob (float): 좌우 반전시킬 확률 값
    """
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # image 좌우 반전
            bbox = target['boxes']
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target
    
    
class ToTensor(object):
    # Transforms 완료된 image의 elements를 텐서로 변환
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
        
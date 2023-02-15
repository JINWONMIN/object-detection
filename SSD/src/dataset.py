"""
@author: Viet Nguyen (nhviet1009@gmail.com)
"""
import torch
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import default_collate
import os


def collate_fn(batch):
    """
    dataset의 sample list를 batch 단위로 바꿔주기 위해 사용.
    """
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items


class CocoDataset(CocoDetection):   # CocoDetection 상속
    def __init__(self, root, year, mode, transform=None):   
        annFile = os.path.join(root, "annotations", "instances_{}{}.json".format(mode, year))   
        root = os.path.join(root, "{}{}".format(mode, year))
        super(CocoDataset, self).__init__(root, annFile)    # root, annFile init
        self._load_categories()     # categories 로드
        self.transform = transform  # 인자로 받은 transform 설정

    def _load_categories(self):

        categories = self.coco.loadCats(self.coco.getCatIds())  # cat id로 카테고리 정보 로드
        categories.sort(key=lambda x: x["id"])  # category id 순으로 sorting

        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"   # background label 추가
        for c in categories:        
            self.label_map[c["id"]] = counter   # category id 재설정
            self.label_info[counter] = c["name"]    # category id의 name 설정
            counter += 1

    def __getitem__(self, item):
        image, target = super(CocoDataset, self).__getitem__(item)  
        width, height = image.size
        boxes = []
        labels = []
        if len(target) == 0:    # annotation이 존재하지 않으면 None
            return None, None, None, None, None
        for annotation in target:
            bbox = annotation.get("bbox")   # box 정보 할당
            boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])    # box 좌표 decode (normalize)
            labels.append(self.label_map[annotation.get("category_id")])    # category id 추가
        boxes = torch.tensor(boxes) # bboxes 텐서로 변환
        labels = torch.tensor(labels)   # category id list to tensor
        if self.transform is not None:  # 인자로 받은 transform 이 None 아니라면,
                                        # transform 적용 후 image, img_id, (height, width), boxes, labels 반환
            image, (height, width), boxes, labels = self.transform(image, (height, width), boxes, labels)
        return image, target[0]["image_id"], (height, width), boxes, labels
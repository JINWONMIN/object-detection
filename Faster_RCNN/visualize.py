import coco_names
from config.visualize_option import parse_args

import torch
import torchvision
import cv2
import numpy as np
import sys, random, os
sys.path.append('./')


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)

def main(args):
    input = []
    if args.dataset == 'coco':
        num_classes = 91
        names = coco_names.names
        
    # Model creating
    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](
        num_classes=num_classes,
        pretrained=False
    )
    model = model.cuda()
    
    model.eval()
    
    save = torch.load(args.model_path)
    model.load_state_dict(save['model'])
    src_img = cv2.imread(args.image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img/255.).permute(2, 0, 1).float().cuda()
    input.append(img_tensor)
    out = model(input)
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']

    for idx in range(boxes.shape[0]):
        if scores[idx] >= args.score:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
            cv2.putText(src_img, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    cv2.imshow('result', src_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    if not os.path.isdir('./assets'):
        os.mkdir('./assets')
    cv2.imwrite('assets/test.jpg', img)


if __name__ == "__main__":
    args = parse_args()
    main()

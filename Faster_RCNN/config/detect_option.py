import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')
    parser.add_argument('--model_path', type=str, default='./result/model_19.pth', help='model path')
    parser.add_argument('--image_path', type=str, default='./test.jpg', help='image path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')
    
    args = parser.parse_args()
    
    return args

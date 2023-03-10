"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from torchvision.models.mobilenet import mobilenet_v2, InvertedResidual


# Base 클래스로 weights 초기화와 bbox view를 수행해준다.
class Base(nn.Module):
    def __init__(self):
        super().__init__()  # nn.Module 을 상속

    # weights를 초기화 해준다.
    def init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]   # *args 문법은 모든 길이의 인수 리스트에 접근하는 데 사용됨.
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):    # src = feature들의 합, loc, conf 레이어들 
        ret = []
        for s, l, c in zip(src, loc, conf): # l = 3x3 conv
            # l(s)view(s.size(0), 4, -1): view(batch size, coords, num of anchor box)
            # l(s) = location에 대한 feature를 통과, s.size(0) = batch size를 의미
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.num_classes, -1)))

        locs, confs = list(zip(*ret))   # zip list 후 locs confs 로 구분
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous() # locs [[], []] -> [] , confs [[], []] -> []
        return locs, confs


# ResNet 새로 정의
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


# SSD의 backbone으로 ResNet을 사용.
class SSD(Base):    # Base class를 상속. --> (weights를 초기화 시키기 위해서.)
    def __init__(self, backbone=ResNet(), num_classes=81):
        super().__init__()

        self.feature_extractor = backbone   # backbone 을 feature extractor로 사용. (default=ResNet)
                                            # 논문에서 backbone으로 VGG16을 사용하였으나 backbone은 자유롭게 바꿀 수 있으므로 ResNet 사용해도 무방함.
        self.num_classes = num_classes  # class 개수 설정. (default=81개) --> dataset에 따라 달라짐.   
        self._build_additional_features(self.feature_extractor.out_channels)    # feaure extractor의 out channels를 받아와서, input channel로 사용함.
                                                                                # Extra feature layers 를 뽑아내기 위해서 사용함.
        self.num_defaults = [4, 6, 6, 6, 4, 4]  # default anchor box 개수 설정
        self.loc = []   # coords
        self.conf = []  # confidences

        # Extra Feature Layers
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):  
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))    # nd * default anchors box 개수   
            self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))    # nd * num classes : 각 클래스별 confidece

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self.init_weights()

    def _build_additional_features(self, input_size):
        # feature extractor의 output channels을 input size로 받음.
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(
                zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):   # i layer의 (input size, output size, channels) 값 할당 
            
            # 3 idx layer 전까지는 feature map을 반으로 줄이면서 계산하고
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),   # stride=2, padding=1 로 feature map을 반으로 줄임.
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            # 그 이후에는 feature map을 줄이지 않고 계산한다.
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)    

        self.additional_blocks = nn.ModuleList(self.additional_blocks)


    def forward(self, x):
        x = self.feature_extractor(x)   # backbone 에서 children() 으로 불러온 extractor
        detection_feed = [x]    
        for l in self.additional_blocks:   # additional block 하나를 layer로 놓음.
            x = l(x)    # layer 하나를 통과한 feature x를
            detection_feed.append(x)    # detection feed에 추가
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs


feature_maps = {}


# MobileNetV2 backbone
class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = mobilenet_v2(pretrained=True).features
        self.feature_extractor[14].conv[0][2].register_forward_hook(self.get_activation())

    def get_activation(self):
        def hook(self, input, output):
            feature_maps[0] = output.detach()

        return hook

    def forward(self, x):
        x = self.feature_extractor(x)
        return feature_maps[0], x


def SeperableConv2d(in_channels, out_channels, kernel_size=3):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                  groups=in_channels, padding=padding),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def StackedSeperableConv2d(ls_channels, multiplier):
    out_channels = 6 * multiplier
    layers = [SeperableConv2d(in_channels=in_channels, out_channels=out_channels) for in_channels in ls_channels]
    layers.append(nn.Conv2d(in_channels=ls_channels[-1], out_channels=out_channels, kernel_size=1))
    return nn.ModuleList(layers)


class SSDLite(Base):
    def __init__(self, backbone=MobileNetV2(), num_classes=81, width_mul=1.0):
        super(SSDLite, self).__init__()
        self.feature_extractor = backbone
        self.num_classes = num_classes

        self.additional_blocks = nn.ModuleList([
            InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
        ])
        header_channels = [round(576 * width_mul), 1280, 512, 256, 256, 64]
        self.loc = StackedSeperableConv2d(header_channels, 4)
        self.conf = StackedSeperableConv2d(header_channels, self.num_classes)
        self.init_weights()


    def forward(self, x):
        y, x = self.feature_extractor(x)
        detection_feed = [y, x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs

## SSD: Single Shot MultiBox Detector

---

### Introduction

- **paper**: [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)
- backbone으로 **SSD-ResNet50** 과 **SSDLite-MobileNetV2** 을 사용하는 파이토치 코드.
- apex.amp를 사용하여 학습. (mixed precision training)

---

### Usage

- **Dataset Download**

```
sh cocodataset_download.sh
```

<br/>

- **train**

```
python train.py --data-path /coco --save-folder trained_models --log-path tensorboard/SSD --epochs 65 --batch-size 32 --amp --local_rank 0
```

<br/>

- **evaluate**

```
python evaluate.py --data-path /coco --cls-threshold 0.5 --nms-threshold 0.5 pretrained-model /path/to/pretrained model/.pth --output predictions
```


import utils
import dataset.transforms as T
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from dataset.coco_utils import get_coco, get_coco_kp
from engine import train_one_epoch, evaluate
from dataset.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from config.train_option import parse_args
from config.config import *
import torchvision


def get_dataset(name, image_set, transform):
    paths = {
        "coco": ('/home/mjw/workspace/data/coco/2017', get_coco, 91),
        "coco_kp": ('/datasets01/COCO/022719', get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    args = parse_args()
    if args.output_dir:
        utils.mkdir(args.output_dir)
    utils.init_distributed_mode(args)

    # Data Loading
    print("Loading data")
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True))
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.b)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.b, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.b,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn
    )
    
    # model creating
    print('Creating model')
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=args.pretrained)
    device = torch.device(args.device)                                                              
    model.to(device)

    # Distribute
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    # Parallel
    if args.parallel:
        print('Training parallel')
        model = torch.nn.DataParallel(model).cuda()
        model_without_ddp = model.module
        
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # Resume training
    if args.resume:
        print("Resume training")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return
    
    # Training
    print('Start training')
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training_time {}'.format(total_time_str))


if __name__ == "__main__":
    main()

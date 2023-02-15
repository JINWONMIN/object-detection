"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from tqdm.autonotebook import tqdm
import torch
from pycocotools.cocoeval import COCOeval
from apex import amp


def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, is_amp):
    # nn.Module에서 train time과 eval time에서 수행하는 다른 작업을 수행할 수 있도록
    # switching 하는 함수. Ex) Dropout Layer, BatchNorm Layer etc..
    # model.train() 은 모든 Layer를 켜둔다.
    model.train()
    num_iter_per_epoch = len(train_loader)  # train loader의 길이를 각 에포크의 총 iteration으로 설정
    progress_bar = tqdm(train_loader)   # train loader로 프로그레스 바 설정
    scheduler.step()    # epoch마다 scheduler.step() 
    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):    # img=img features, gloc=gt bbox, glabel=gt category id
        if torch.cuda.is_available():   # cuda가 존재 한다면
            img = img.cuda()    # img를 cuda에 싣는다.
            gloc = gloc.cuda()  # gloc를 cuda에 싣는다.
            glabel = glabel.cuda()  # glabel을 cuda에 싣는다.

        ploc, plabel = model(img)   # model에 img 를 넣어서 prediction loc 와 label을 얻는다.
        ploc, plabel = ploc.float(), plabel.float()     # prediction loc 와 label을 float로 변환
        gloc = gloc.transpose(1, 2).contiguous()        # 1, 2 차원의 위치를 바꾼 후 메모리르 재 할당 해준다.
        loss = criterion(ploc, plabel, gloc, glabel)    # loss 계산을 위해 prediction 값과 ground truth 값을 넣어준다.

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))  # 현재 epoch와 계산한 loss를 프린트 해준다.

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)    # scalar를 tensorboard로 넣어준다. (loss and epoch)

        """
        amp(Automatic Mixed Precision)는 apex 에서 단 3줄만으로 mixed precision으로 학습할 수 있게 만들어주는 도구.
        mixed precision 학습을 통해 배치 사이즈 증가, 학습시간 단축, 성능 증가의 장점을 얻을 수 있음.
        
        Mixed Precision: 처리 속도를 높이기 위해 FP16(16bit Floating Point) 연산과 FP32(32bit Floating Point)를 섞어서 학습하는 방법.
                         FP32 대신 FP16을 사용하게 되면 절반의 메모리 사용량과 8배의 연산 처리량의 이점을 가짐.
        """
        if is_amp:  # 만약 amp가 세팅되어 있다면
            with amp.scale_loss(loss, optimizer) as scale_loss: # 학습 중에 loss와 optimizer를 amp.scale_loss로 감싼다.
                scale_loss.backward()   # 감싼 scaled_loss로 back propagation을 진행
        else:   # 아니라면
            loss.backward() # 기존의 loss로 back propagation 진행
        optimizer.step()    # optimizer update
        optimizer.zero_grad()   # optimizer 초기화 (Pytorch 에서는 gradients 값 들을 추후에 backward를 해줄 때 계속 더해 주기 때문에 backward 전 0으로 만들어준다.)


def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold):
    model.eval()    # model.eval()로 switching   --> Dropout layer, BatchNorm Layer 등을 off로 전환
    detections = []
    category_ids = test_loader.dataset.coco.getCatIds() # test_loader의 category id를 불러온다. (pycocotools의 getCatIds() 함수로 쉽게 로드할 수 있음.)
    for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):    # 학습을 할게 아니기 때문에 img, img_id, img_size만 받는다. (gt bbox, category id는 필요 없음.)
        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")    # 현재 배치 / 총 길이
        if torch.cuda.is_available():   # cuda가 있다면
            img = img.cuda()    # img를 cuda로 올린다.
        with torch.no_grad():   # with torch.no_grad()를 해주어 autograd engine을 비활성화 시켜 필요한 메모리를 줄여주고 연산속도를 증가시킨다. 
                                # dropout을 비활성화 시키지는 않음. model.eval()과는 다름.
            # Get predictions
            ploc, plabel = model(img)   # prediction 한 정보를 얻는다.
            ploc, plabel = ploc.float(), plabel.float() # prediction한 loc와 label을 float로 변환

            for idx in range(ploc.shape[0]):    
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                                       (loc_[3] - loc_[1]) * height, prob_,
                                       category_ids[label_ - 1]])

    detections = np.array(detections, dtype=np.float32)

    coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)

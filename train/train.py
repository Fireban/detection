# -*- coding: utf-8-*-
from __future__ import division

import sys
import os
import time
import datetime
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from serverDetect.models import *
from utils.logger import *
from serverDetect.utils.utils import *
from utils.datasets import *
from serverDetect.utils.parse_config import *
from test import evaluate
from terminaltables import AsciiTable
from torch.autograd import Variable
from trainConfig.trainConfig import trainConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="yolo", help="which model to train 'yolo' / 'yolotiny'")
    opt = parser.parse_args()

    config = trainConfig(type=opt.type)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs('result_save', exist_ok=True)

    # 데이터 관련 경로, 클래스 명 가져오기
    data_config = parse_data_config(config.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # 모델 구조 형성 & 초기화
    model = Darknet(config.model_def).to(device)
    model.apply(weights_init_normal)

    # 저장된 가중치가 있다면 읽어와서 모델에 적용
    if config.pretrained_weights:
        if config.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(config.pretrained_weights, map_location=device))
        else:
            model.load_darknet_weights(config.pretrained_weights)

    #
    dataset = ListDataset(train_path, augment=True, multiscale=config.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # epochs만큼 학습 시작
    for epoch in range(config.epochs):
        # 모델 가중치가 갱신 되도록 전환
        model.train()
        start_time = time.time()
        # batch size 단위로 이미지와 label을 읽어온다.
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            # 현재 학습 환경에 맞게 데이터 타입 변경
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # 모델에 이미지와 라벨을 입력하고 손실과 결과 값 받아오기.
            loss, outputs = model(imgs, targets)
            # 역전파할 손실을 계산하여 모델 가중치 변화도 계산 & 누적
            loss.backward()

            if batches_done % config.gradient_accumulations:
                # 역전파로 계산된 모델의 가중치 변화도 만큼 가중치 갱신
                optimizer.step()
                # 누적된 값 지우기
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, config.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % config.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # validation 성능 평가
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=config.img_size,
                batch_size=1,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            # 과거의 최소 validataion loss 보다 더 작은 loss면 모델을 저장하고 minvalidation을 갱신
            if AP.mean() < config.min_val_loss:
                torch.save(model.state_dict(), config.save_path)
                with open(config.min_val_loss_path, 'w') as f:
                    f.write(str(AP.mean()))

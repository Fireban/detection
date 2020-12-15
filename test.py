#-*- coding: utf-8-*-
from __future__ import division

from yolo.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import *
from yolo.utils.parse_config import *

import os
import argparse
import tqdm
import numpy as np
from PIL import ImageDraw

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, save=True):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        gray_result = transforms.ToPILImage()(imgs.squeeze()).convert("RGB")

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        try:
            detections = outputs[0].numpy()
        except:
            continue


        for d_num, (x_min, y_min, x_max, y_max, obj_conf, cls_conf, cls_num) in enumerate(detections):

            if cls_num != 0:
                continue

            x_min, y_min = x_min / img_size, y_min / img_size
            x_max, y_max = x_max / img_size, y_max / img_size
            x_min = x_min * img_size
            y_min = y_min * img_size
            x_max = x_max * img_size
            y_max = y_max * img_size
            width = x_max - x_min
            height = y_max - y_min

            if save:
                draw = ImageDraw.Draw(gray_result)
                draw.rectangle([(x_min, y_min), (x_min + width, y_min + height)],
                               outline="red")
        if save:
            gray_result.save(os.getcwd()+'/result_save/'+str(batch_i)+'.png')

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default=None, help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/CUSTOM/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path != None:
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path, map_location=device))

    print("Compute mAP...")
    os.makedirs('result_save', exist_ok=True)

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=1,
        save=True,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

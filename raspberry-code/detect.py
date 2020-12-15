from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from torch.utils.data import DataLoader
from post import postSender
from PIL import Image
from torch.autograd import Variable

import os
import time
import argparse
import uuid
import torch
import torchvision.transforms as transforms

frame_width = 320
frame_height = 240
frame_pixel_byte = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="test_images", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/weights_for_run.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    post_uri = 'http://www.fireban.kr/api/detect/find/'
    mac = uuid.getnode()
    img_width = frame_width
    img_height = frame_height

    # Post sender (URL, mac, img_width, img_height)
    postSender = postSender(post_uri, mac, img_width, img_height)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights_path, map_location=device))

    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    while True:
        time.sleep(0.5)
        prev_time = time.time()
        with open('/dev/video4', 'rb') as v:
            img_data = v.read(frame_width * frame_height * frame_pixel_byte)
            img = Image.frombytes('RGBA', (frame_width, frame_height), img_data, 'raw')
            img_tensor = transforms.ToTensor()(img)[:3, :, :]
            img_tensor, _ = pad_to_square(img_tensor, 0)
            img_tensor = torch.unsqueeze(img_tensor, 0)
            # Configure input
            img_tensor = Variable(img_tensor.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(img_tensor)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            print('Done!')

            # Save image and detections
            print('detections: ', detections)
            if len(detections) != 0 and type(detections[0]) == torch.Tensor:
                print(postSender.sendMessage(img_data, detections[0]))
                # image post
        print("processing time ", time.time() - prev_time)

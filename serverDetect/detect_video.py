#-*- coding: utf-8-*-
import os
import torchvision.transforms as transforms
import cv2

from .models import *
from .utils.utils import *
from PIL import Image, ImageDraw
from .db_models import *
from datetime import datetime
from .ESPCN.espcn import ESPCN
from torchvision.transforms import ToTensor



class VideoDetect():
    def __init__(self, config):
        self.model = None
        self.model_def = config.model_def
        self.lr_to_hr_model = None
        self.lr_to_hr_ratio = config.lr_to_hr_ratio
        self.lr_to_hr_weights_path = config.lr_to_hr_weights_path
        self.weights_path = config.weights_path
        self.label = config.class_path
        self.conf_thres = config.conf_thres
        self.nms_thres = config.nms_thres
        self.batch_size = config.batch_size
        self.img_size = (config.img_size, config.img_size) if type(config.img_size) != tuple else config.img_size
        self.save = config.save
        self.video_path = config.video_path
        self.out = None
        self.out_frames = []
        self.out_fps = 2
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        self.classes = load_classes(config.class_path)
        self.result_image_dir = config.result_image_dir
        self.origin_image_dir = config.origin_image_dir
        self.image_transforms = transforms.Compose([transforms.Resize(size=self.img_size),
                                            transforms.ToTensor()])
        self.hw_key = config.hw_key

        os.makedirs(self.result_image_dir, exist_ok=True, mode=0o777)
        os.makedirs(self.origin_image_dir, exist_ok=True, mode=0o777)

    def load_model(self):
        # Set model
        self.model = Darknet(self.model_def, img_size=self.img_size)
        self.model.to(self.device)
        self.lr_to_hr_model = ESPCN(self.lr_to_hr_ratio)

        # Loading the checkpoint weights
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()
        self.lr_to_hr_model.load_state_dict(torch.load(self.lr_to_hr_weights_path, map_location=self.device))
        self.lr_to_hr_model.eval()

    def run_detect(self):
        self.load_model()
        print('Performing Object Detection\n')

        cap = cv2.VideoCapture(self.video_path)

        while True:
            frame_tic = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            db_detect = []

            ret, fram = cap.read()

            is_detect = False
            print('working...')

            if not ret:
                print('End')
                break

            orig_image = Image.fromarray(np.uint8(fram))
            orig_image_size = orig_image.size


            gray_origin = Image.fromarray(cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY))
            gray_result = gray_origin.copy()

            y, cb, cr = gray_origin.convert('YCbCr').split()
            img = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0]).to(self.device)

            with torch.no_grad():
                img = self.lr_to_hr_model(img)
                img = F.interpolate(img, min(img.shape[-1], img.shape[-2]))
                img = img.expand(-1, 3, -1, -1)

                detections = self.model(img)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

            try:
                detections = detections[0].numpy()
                # The detections contain 7 columns which are [x_min, y_min, x_max, y_max, obj_conf, cls_conf, cls_num]
                # and the bounding box coordinates are in 'opt.img_size' scale
                unique_classes = np.unique(detections[:, -1])
                n_detections = detections.shape[0]
                colors = np.random.rand(n_detections, 3)

                for d_num, (x_min, y_min, x_max, y_max, obj_conf, cls_conf, cls_num) in enumerate(detections):

                    if cls_num != 0:
                        continue

                    x_min, y_min = x_min / self.img_size[0], y_min / self.img_size[1]
                    x_max, y_max = x_max / self.img_size[0], y_max / self.img_size[1]
                    x_min = x_min * orig_image_size[0]
                    y_min = y_min * orig_image_size[1]
                    x_max = x_max * orig_image_size[0]
                    y_max = y_max * orig_image_size[1]
                    width = x_max - x_min
                    height = y_max - y_min
                    cls_num = int(cls_num)

                    name = self.classes[cls_num]

                    print(cls_num, x_min, y_min, x_max, y_max)
                    print(x_max - x_min)
                    print(y_max - y_min)

                    if width < 60 and height < 60:
                        is_detect = True

                        if self.save:
                            draw = ImageDraw.Draw(gray_result)
                            draw.rectangle([(x_min, y_min), (x_min+width, y_min+height)], outline="red")

                            db_detect.append(DetectTargetdetection(cls_num, x_min, y_min, x_max, y_max,
                                                                   frame_tic,None))

            except:
                pass


            if self.save and is_detect:
                origin_image_path = os.path.join(self.origin_image_dir, frame_tic+'.png')
                gray_origin.save(origin_image_path)

                hw_id = HwInfoProduct.query.filter_by(authKey=self.hw_key).first().id

                db_image_path = origin_image_path.split('/')
                db_image_path = '/'.join(db_image_path[3:])
                db_image = DetectTargetimage(db_image_path, frame_tic, hw_id)
                db.session.add(db_image)

                current_image = DetectTargetimage.query.filter_by(createAt=frame_tic, target_id = hw_id).first()
                current_image_id = current_image.id

                result_image_path = os.path.join(self.result_image_dir, frame_tic+'.png')
                gray_result.save(result_image_path)

                for db_detect in db_detect:
                    db_detect.targetImage_id = current_image_id
                    db.session.add(db_detect)

                db.session.commit()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_key = 'testing_key'
    detect_start_time = time.time()

    model_def = "config/yolov3-custom_flir.cfg"
    weights_path = "checkpoints/model_param_flir.pth"
    class_path = "data/classes.names"
    conf_thres = 0.5
    nms_thres = 0.3
    batch_size = 1
    img_size = 320
    save = True
    cuda = torch.cuda.is_available()


    class _Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)


    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True, mode=0o777)

    config = {
        'model_def': os.path.join(base_dir, model_def),
        'weights_path': os.path.join(base_dir, weights_path),
        'class_path': os.path.join(base_dir, class_path),
        'result_image_dir': os.path.join(output_dir, 'result', detect_key, str(detect_start_time)),
        'origin_image_dir': os.path.join(output_dir, 'origin', detect_key, str(detect_start_time)),
        'conf_thres': conf_thres,
        'nms_thres': nms_thres,
        'batch_size': batch_size,
        'img_size': img_size,
        'save': save,
        'cuda': cuda,
        'video_path': None,
    }
    args = _Args(**config)

    args.video_path = os.path.join(base_dir, 'test_image_dir')

    solver = VideoDetect(args)
    solver.detect_image_folder()

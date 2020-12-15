# -*- coding: utf-8-*-
import os
import time

from .models import *
from .utils.utils import *
from PIL import Image, ImageDraw
from .db_models import *
from datetime import datetime
from .ESPCN.espcn import ESPCN
from torchvision.transforms import ToTensor
from torch.autograd import Variable


class ImageDetect():
    def __init__(self, config):
        # 파라메터 설정
        self.model = None
        self.lr_to_hr_model = None
        self.model_def = config.model_def
        self.model_weights_path = config.weights_path
        self.lr_to_hr_ratio = config.lr_to_hr_ratio
        self.lr_to_hr_weights_path = config.lr_to_hr_weights_path
        self.label = config.class_path
        self.conf_thres = config.conf_thres
        self.nms_thres = config.nms_thres
        self.batch_size = config.batch_size
        self.img_size = (config.img_size, config.img_size)
        self.device = config.cuda
        self.classes = load_classes(config.class_path)
        self.result_image_dir = config.result_image_dir
        self.origin_image_dir = config.origin_image_dir
        self.hw_key = config.hw_key

        # 결과 & 원본 이미지 저장 디렉토리 생성
        os.makedirs(self.result_image_dir, exist_ok=True, mode=0o777)
        os.makedirs(self.origin_image_dir, exist_ok=True, mode=0o777)

    def load_model(self):
        # 탐지 & 고해상화 모델 구조 load
        self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device) # debug self.model.to(self.device)
        self.lr_to_hr_model = ESPCN(self.lr_to_hr_ratio).to(self.device)

        # 탐지 & 고해상화 모델 가중치 load
        self.model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
        self.lr_to_hr_model.load_state_dict(torch.load(self.lr_to_hr_weights_path, map_location=self.device))

        # 모델 가중치 고정
        self.model.eval()
        self.lr_to_hr_model.eval()

    # 탐지된 정보를 이미지 경로와 함께 DB에 저장
    def insert_to_db(self, origin_image_path, frame_tic, db_detect):
        # hw key로 hw id 가져오기
        hw_id = HwInfoProduct.query.filter_by(authKey=self.hw_key).first().id

        # 이미지 경로 DB에 저장
        db_image_path = origin_image_path.split('/')
        db_image_path = '/'.join(db_image_path[3:]) # DB에 저장하기 전에 /var/www 없이 저장해야합니다(우석님 요청.)
        db_image = DetectTargetimage(db_image_path, frame_tic, hw_id)
        db.session.add(db_image)

        # 방금 저장한 이미지의 id 추출
        current_image = DetectTargetimage.query.filter_by(createAt=frame_tic, target_id=hw_id).first()
        current_image_id = current_image.id

        # 탐지된 좌표에 해당 이미지 id값 추가하여 DB에 저장
        for cls_num, x_min, y_min, x_max, y_max, frame_tic, _ in db_detect:
            detect_target_detection = DetectTargetdetection(cls_num, x_min, y_min, x_max,y_max,
                                                            frame_tic, current_image_id)
            db.session.add(detect_target_detection)

        # DB 트렌젝션 종료
        db.session.commit()

    # 최종 탐지 좌표를 원래의 이미지에서의 값으로 변환
    def calc_origin_points(self, size_origin, size_result, x_min, y_min, x_max, y_max):

        # (원본 이미지 길이 / 결과 이미지 길이) * 원본 좌표
        x_min = int((size_origin[0] / size_result[0]) * x_min)
        y_min = int((size_origin[1] / size_result[1]) * y_min)
        x_max = int((size_origin[0] / size_result[0]) * x_max)
        y_max = int((size_origin[1] / size_result[1]) * y_max)

        return x_min, y_min, x_max, y_max

    # 지정 폴더에 원본 & 결과 이미지 저장, 원본 이미지 경로 반환
    def save_image(self, frame_tic, image_origin, image_size_origin, image_result):
        # 원본 이미지 저장
        origin_image_path = os.path.join(self.origin_image_dir, frame_tic + '.png')
        image_origin.resize(image_size_origin)
        image_origin.save(origin_image_path)

        # 결과 이미지 저장
        result_image_path = os.path.join(self.result_image_dir, frame_tic + '.png')
        image_result.save(result_image_path)

        return origin_image_path


    # 서버에 저장되는 이미지에 대해 탐지 시작.
    def run_detect(self, image_path):
        # 이미지 불러오기
        image_origin = Image.open(image_path)
        # 모델 load
        self.load_model()
        # 이미지 저장 이름을 위해 현재 시간 소수점까지 가져오기.
        frame_tic = datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
        # 탐지된 좌표 저장 list
        db_detect = []
        # 탐지가 되었을 때만 이미지를 저장하기 위한 변수
        is_detect = False
        # 원본 이미지 사이즈 추출
        image_size_origin = image_origin.size
        image_result = image_origin.copy()

        # 0~1사이의 gray scale로 이미지 추출
        y, cb, cr = image_origin.convert('YCbCr').split()
        img = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0]).to(self.device)

        # with scope 안에서는 역전파를 위한 기울기를 구하지 않음.
        with torch.no_grad():
            # 이미지 실시간 고해상화 진행
            img = self.lr_to_hr_model(img)
            # 정사각형으로 이미지 변환 ex) (480, 640) -> (640, 640)
            img = F.interpolate(img, max(img.shape[-1], img.shape[-2]))
            # 1 channel -> 3 channel로 변환 # later (추후 제거 가능: model train 시에 config 변경)
            img = img.expand(-1, 3, -1, -1)
            # 마지막 이미지 크기 추출
            image_size_result = img.shape[-2:]

            # 이미지에 대한 탐지 진행 [x_min, y_min, x_max, y_max, obj_conf, cls_conf, cls_num]
            detections = self.model(img)
            # 탐지된 좌표에 대해 NMS 적용
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

        try:
            # 탐지 좌표 자료형 변환
            detections = detections[0].numpy()

            for d_num, (x_min, y_min, x_max, y_max, obj_conf, cls_conf, cls_num) in enumerate(detections):
                # 탐지된 객체가 있음.
                is_detect = True
                # 원본 이미지 사이즈의 좌표로 변환
                x_min, y_min, x_max, y_max = self.calc_origin_points(image_size_origin, image_size_result,
                                                                     x_min, y_min, x_max, y_max)

                # 탐지 결과 시각화 # later (추후 제거 가능: 결과 이미지를 볼 필요 없을때)
                draw = ImageDraw.Draw(image_result)
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red")

                # 탐지 좌표 저장
                db_detect.append((cls_num, x_min, y_min, x_max, y_max, frame_tic, None))

        except Exception as e:
            print(detections)
            print(e)

        # 해당 이미지에서 객체가 있다면 저장(이미지 & DB).
        if is_detect:
            # 이미지 저장
            origin_image_path = self.save_image(frame_tic, image_origin, image_size_origin, image_result)

            # DB에 원본 이미지 경로 & 탐지된 좌표들 저장
            self.insert_to_db(origin_image_path, frame_tic, db_detect)

# 테스트용 코드
if __name__ == '__main__':
    detect_key = 'testing_key'
    detect_start_time = time.time()

    model_def = "config/yolov3.cfg"
    weights_path = "weights/weights_for_run.pth"
    class_path = "data/classes.names"
    conf_thres = 0.3
    nms_thres = 0.3
    batch_size = 1
    img_size = 640
    cuda = torch.cuda.is_available()


    class _Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)


    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')

    config = {
        'model_def': os.path.join(base_dir, model_def),
        'weights_path': os.path.join(base_dir, weights_path),
        'class_path': os.path.join(base_dir, class_path),
        'result_image_dir': os.path.join(output_dir, 'result', detect_key),
        'origin_image_dir': os.path.join(output_dir, 'origin', 'images'),
        'conf_thres': conf_thres,
        'nms_thres': nms_thres,
        'batch_size': batch_size,
        'img_size': img_size,
        'cuda': cuda,
        'image_path': os.path.join(base_dir, 'test_image_dir', '10m_15m_499.png'),
        'lr_to_hr_ratio': 2,
        'lr_to_hr_weights_path': 'ESPCN/weights/weights.pt',
        'hw_key': detect_key,
    }
    args = _Args(**config)

    os.makedirs(output_dir, exist_ok=True, mode=0o777)

    solver = ImageDetect(args)
    solver.run_detect(args.image_path)

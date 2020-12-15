#-*- coding: utf-8-*-
import os
import torch
import time
import json

from flask import Flask
from flask import request
from celery import Celery

from detect_celery import get_image_path
from yolo.detect_images import ImageDetect
from yolo.db_models import db

''' redis, celery 켜기
redis-server
celery -A app.celery worker --loglevel=info
'''


def make_celery(app):
    celery = Celery(
        'app',
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


flask_app = Flask(__name__)
flask_app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379',
    SQLALCHEMY_DATABASE_URI = 'mysql://fireban:fireban12#$@localhost/fireban',
)
celery = make_celery(flask_app)
db.init_app(flask_app)


def on_json_loading_failed_return_dict(e):
    return {}


@celery.task()
def start_detection(data_dir, out_dir, detect_key, detect_start_time):

    print("celery start!")
    # 탐지에 필요한 설정 값들
    model_def = "yolo/config/yolov3.cfg"
    weights_path = "yolo/weights/weights_for_run.pth"
    class_path = "yolo/data/classes.names"
    conf_thres = 0.4
    nms_thres = 0.3
    batch_size = 1
    img_size = 640
    save = True
    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_to_hr_ratio = 2
    lr_to_hr_weights_path = "yolo/ESPCN/weights/weights.pt"

    class _Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = out_dir
    os.makedirs(data_dir, exist_ok=True, mode=0o777)
    os.makedirs(output_dir, exist_ok=True, mode=0o777)

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
        'save': save,
        'cuda': cuda,
        'hw_key': detect_key,
        'lr_to_hr_ratio': lr_to_hr_ratio,
        'lr_to_hr_weights_path': lr_to_hr_weights_path,
    }
    args = _Args(**config)

    # 해당 경로의 영상을 시간 순으로 하나씩 읽어와서 탐지한다.

    solver = ImageDetect(args)
    for image_path in get_image_path(data_dir, detect_key, detect_start_time):
        solver.run_detect(image_path)
        os.remove(image_path)

    return True


@flask_app.route('/start_detect', methods=['POST'])
def start_detect():
    request.on_json_loading_failed = on_json_loading_failed_return_dict
    json_data = request.get_json(silent=True)

    if json_data != {}:
        try:
            # 이미지 경로, 드론 키값, 시작 시간
            key = json_data['hw_key']
            start_time = time.time()
            start_time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            data = time.strftime('%Y%m%d', time.localtime(start_time))

            path = '/var/www/detect/' + data
            output_dir = '/var/www/output'

            # detection 비동기 시작
            start_detection.delay(data_dir=path, out_dir=output_dir, detect_key=key, detect_start_time=start_time_str)
        except Exception as e:
            print(e)
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}


if __name__ == '__main__':
    flask_app.run()

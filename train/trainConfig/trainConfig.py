import os


class trainConfig():
    def __init__(self, type = 'yolo'):
        # 학습 반복 에폭수 (전체 데이터 한번 학습 완료 = 1 epoch)
        self.epochs = 1000
        # 한번의 학습에 사용할 이미지 수(dataloader에서 한번에 몇 장의 이미지가 나올지)
        self.batch_size = 8
        # 몇개의 배치 수를 학습하고 역전파 값으로 가중치를 갱신할지
        self.gradient_accumulations = 2
        # 읽어 올 데이터 위치 적혀있음.
        self.data_config = "../serverDetect/config/custom.data"
        # 이미지 읽어오거나 할때 사용할 cpu 코어 수 지정
        self.n_cpu = 1
        # 모델 가중치 저장할 시점들(checkpoint_interval의 배수 epoch 마다 저장함.)
        self.checkpoint_interval = 1
        # validation 구할 횟수
        self.evaluation_interval = 1
        # mAP 지표 10번째 배치때 계산 할지
        self.compute_map = False
        # 인풋 이미지를 멀티 스케일 해서 학습 할지
        self.multiscale_training = True

        # 입력 이미지 크기
        self.img_size = None
        # 이미 저장되어있는 모델 가중치 경로
        self.pretrained_weights = None
        # 모델 구조 저장 파일 경로
        self.model_def = None
        # 어디에 다시 모델 저장할지
        self.save_path = None
        # 그동안의 학습에서 가장 작았던 validation loss를 저장 해놓을 경로
        self.min_val_loss_path = 'trainConfig/minValidation_'+type
        # 그동안의 학습에서 가장 작았던 validation loss 값
        self.min_val_loss = float(open(self.min_val_loss_path).read())

        # yolo 학습이나 yolo tiny 학습이냐에 따라 달라지는 값.
        if type == 'yolotiny':
            self.img_size = 320
            self.model_def = "../raspberry-code/config/yolov3-tiny.cfg"
            self.pretrained_weights = "../raspberry-code/weights/weights_for_run.pth"
            self.save_path = "../raspberry-code/weights/weights_for_run.pth"
        elif type == 'yolo':
            self.img_size = 640
            self.model_def = "../serverDetect/config/yolov3.cfg"
            self.pretrained_weights = "../serverDetect/weights/weights_for_run.pth"
            self.save_path = "../serverDetect/weights/weights_for_run.pth"

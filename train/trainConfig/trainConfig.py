
class trainConfig():
    def __init__(self, type = 'yolo'):
        # number of epochs
        self.epochs = 100
        # size of each image batch
        self.batch_size = 8
        # number of gradient accums before step
        self.gradient_accumulations = 2
        # path to data config file
        self.data_config = "../serverDetect/config/custom.data"
        # number of cpu threads to use during batch generation
        self.n_cpu = 1
        # interval between saving model weights
        self.checkpoint_interval = 1
        # interval evaluations on validation set
        self.evaluation_interval = 1
        # if True computes mAP every tenth batch
        self.compute_map = False
        # allow for multi-scale training
        self.multiscale_training = True

        # size of each image dimension
        self.img_size = None
        # if specified starts from checkpoint model
        self.pretrained_weights = None
        # path to model definition file
        self.model_def = None
        # where to save model
        self.save_path = None

        if type == 'yolotiny':
            self.img_size = 320
            self.model_def = "../raspberry-code/config/yolov3-tiny.cfg"
            self.pretrained_weights = "../raspberry-code/weights/weights_for_run.pth"
            self.save_path = "../raspberry-code/weights/weights_for_ru.pth"
        elif type == 'yolo':
            self.img_size = 640
            self.model_def = "../serverDetect/config/yolov3.cfg"
            self.pretrained_weights = "../serverDetect/weights/weights_for_run.pth"
            self.save_path = "../serverDetect/weights/weights_for_ru.pth"
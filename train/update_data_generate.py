#-*- coding: utf-8-*-
import pymysql
import os
from PIL import Image


def image_size(path):
    img = Image.open(path)
    w, h = img.size

    return w, h


class trainInfoGenerator():
    def __init__(self, user, passwd, host, db, image_dir, label_dir):
        self.fireban_db = pymysql.connect(
            user=user,
            passwd=passwd,
            host=host,
            db=db,
        )
        self.cursor = self.fireban_db.cursor(pymysql.cursors.DictCursor)
        # DB에서 탐지된 정보 가져오는 sql query
        self.label_sql = "select detect.detectType, detect.xmin, detect.ymin, detect.xmax, detect.ymax, target.path \
                            from detect_targetdetection as detect \
                            JOIN detect_targetimage as target on target.id= detect.targetImage_id \
                            WHERE detectType=0;"
        # DB에서 탐지된 이미지 리스트 가져오는 sql query
        self.train_sql = "SELECT path FROM `detect_targetimage` where isUpdated=1;"
        self.image_dir = image_dir
        self.label_dir = label_dir

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)

    # DB에서 이미지에 대한 탐지 정보를 가져와서 label txt를 만들어준다.
    def label_txt_generate(self):
        self.cursor.execute(self.label_sql)
        result = self.cursor.fetchall()

        for i in result:
            filename = str(i['path'].split('/')[-1].replace('.png', ''))
            image_png_path = '/var/www' + i['path']
            label_txt_path = os.path.join(self.label_dir, filename+'.txt')
            w, h = image_size(image_png_path)

            x_center = (int(i['xmin']) + int(i['xmax'])) / h
            y_center = (int(i['ymin']) + int(i['ymax'])) / w
            width = (int(i['xmax']) - int(i['xmin'])) / h
            height = (int(i['ymax']) - int(i['ymin'])) / w

            f = open(label_txt_path, mode='w')
            f.write(str(i['detectType']) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(
                width) + ' ' + str(height) + '\n')
            f.close()

    # train 할 이미지의 리스트를 만들어준다.
    def train_txt_generate(self):
        self.cursor.execute(self.train_sql)
        result = self.cursor.fetchall()

        f = open('../serverDetect/data/train.txt', 'a')
        for i in result:
            if(len(i['path']) > 2):
                f.write('/var/www' + i['path'] + '\n')
        f.close()

    def run(self):
        self.train_txt_generate()
        self.label_txt_generate()


if __name__ == '__main__':
    train_info_generator = trainInfoGenerator('fireban', 'fireban12#$', '127.0.0.1', 'fireban', '/var/www/output/origin/images', '/var/www/output/origin/labels')
    train_info_generator.run()

#-*- coding: utf-8-*-
import pymysql
import os

class trainInfoGenerator():
    def __init__(self, user, passwd, host, db, image_dir, label_dir):
        self.fireban_db = pymysql.connect(
            user=user,
            passwd=passwd,
            host=host,
            db=db,
        )
        self.cursor = self.fireban_db.cursor(pymysql.cursors.DictCursor)
        self.label_sql = "select detect.detectType, detect.xmin, detect.ymin, detect.xmax, detect.ymax, target.path \
                            from detect_targetdetection as detect \
                            JOIN detect_targetimage as target on target.id= detect.targetImage_id;"
        self.train_sql = "SELECT path FROM `detect_targetimage`;"
        self.image_dir = image_dir
        self.label_dir = label_dir

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)

    def label_txt_generate(self):
        self.cursor.execute(self.label_sql)
        result = self.cursor.fetchall()

        for i in result:
            filename = str(i['path'].split('/')[-1].replace('.png', ''))
            label_txt_path = os.path.join(self.label_dir, filename+'.txt')

            x_center = (int(i['xmin']) + int(i['xmax'])) / 240
            y_center = (int(i['ymin']) + int(i['ymax'])) / 320
            width = (int(i['xmax']) - int(i['xmin'])) / 240
            height = (int(i['ymax']) - int(i['ymin'])) / 320

            f = open(label_txt_path, mode='w')
            f.write(str(i['detectType']) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(
                width) + ' ' + str(height) + '\n')
            f.close()

    def train_txt_generate(self):
        self.cursor.execute(self.train_sql)
        result = self.cursor.fetchall()

        #f = open('./' + datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.txt', mode='w')
        f = open('yolo/data/train.txt', 'w')
        for i in result:
            if(len(i['path']) > 2):
                f.write('/var/www/' + i['path'] + '\n')
        f.close()

    def run(self):
        self.train_txt_generate()
        self.label_txt_generate()


if __name__ == '__main__':
    train_info_generator = trainInfoGenerator('fireban', 'fireban12#$', '127.0.0.1', 'fireban', '/var/www/output/origin/image', '/var/www/output/origin/labels')
    train_info_generator.run()

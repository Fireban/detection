import os
import glob
import xml.etree.ElementTree as ET
import argparse

def ret_label(label_string):
    if label_string == 'person':
        return 0
    else:
        return None

def xml_to_txt(xml_path, label_dir):

    for xml_file in glob.glob(xml_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        file_name = xml_file.split('/')[-1].replace('.xml', '')
        img_width = int(root.find('size')[0].text)
        img_height = int(root.find('size')[1].text)
        label_path = os.path.join(label_dir, file_name + ".txt")

        if os.path.exists(label_path):
            os.remove(label_path)

        with open(label_path, 'a') as f:
            for member in root.findall('object'):

                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)

                xcenter = (xmin + xmax) / 2
                ycenter = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                f.write(str(0) + ' ')
                f.write(str(xcenter / img_width) + ' ')
                f.write(str(ycenter / img_height) + ' ')
                f.write(str(width / img_width) + ' ')
                f.write(str(height / img_height) + '\n')

        yield file_name + '.png'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_path", default="/var/www/output/origin", help="data root directory path")
    parser.add_argument("--save_dir", default=os.path.join(os.getcwd(), '../serverDetect/data'), help="txt save path")
    opt = parser.parse_args()
    print(opt)

    for folder in ['train', 'valid']:
        xml_dir = os.path.join(opt.data_root_path, 'xml_data', folder)
        image_dir = os.path.join(opt.data_root_path, 'images')
        label_save_dir = os.path.join(opt.data_root_path, 'labels')
        image_txt_path = os.path.join(opt.save_dir, folder + '.txt')

        os.makedirs(label_save_dir, exist_ok=True)

        if os.path.exists(image_txt_path):
            os.remove(image_txt_path)

        for img_name in xml_to_txt(xml_dir, label_save_dir):
            with open(image_txt_path, 'a') as f:
                image_info = os.path.join(image_dir, img_name)
                f.write(image_info + '\n')

    print('Successfully converted xml to txt.')

#-*- coding: utf-8-*-
import os
import glob
import natsort
import time
from serverDetect.db_models import *


def get_image_path(image_dir, image_key, detect_start_time):
    # 비디오 검색 경로 지정
    image_path = os.path.join(image_dir, image_key + '*')

    while True:
        # 해당 HW가 종료되면 탐지를 종료함.
        db.session.commit()
        hw_info = db.session.query(StreamInfoStream).filter(StreamInfoStream.key==image_key).first()
        hw_is_active = hw_info.isActive
        # 타임 딜레이
        time.sleep(20)

        # 디렉토리에 존재하는 Key로 시작하는 모든 사진 읽어오기
        file_list = glob.glob(image_path)
        # png 파일만 읽어오기
        image_list = [image for image in file_list if image.endswith(".png")]
        # 파일 이름이 시간으로 저장되므로 존재하는 파일들을 시간순으로 정렬해준다.
        image_list = natsort.natsorted(image_list)

        # start time보다 전에 있는 영상은 제외
        recent_image_list = [recent_image for recent_image in image_list if
                             recent_image.split('-')[2] >= detect_start_time]
        # 동영상이 존재하지않을 때
        if len(recent_image_list) == 0:
            # 이미지가 없을 경우
            # 이미지를 다시 찾기위해 continue.
            print("waiting image...")
            if hw_is_active == 0:
                break
            else:
                continue

        # detect할 동영상 파일 경로를 하나씩 반환해준다.
        for recent_image_path in recent_image_list:
            yield recent_image_path

import requests
import base64


class postSender():
    def __init__(self, URL, mac, img_width, img_height):
        self.URL = URL
        self.mac = mac
        self.img_width = img_width
        self.img_height = img_height

    def sendMessage(self, img_bytes, detections):
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[0]
        data = {
            "mac": self.mac,
            "image": img_base64,
            "width": self.img_width,
            "height": self.img_height,
            "min_x": int(x1),
            "min_y": int(y1),
            "max_x": int(x2),
            "max_y": int(y2),
            "type": str(0)
        }
        response = requests.post(self.URL, data=data)
        return response

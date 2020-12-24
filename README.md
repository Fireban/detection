# swmaestro-fireban
소프트웨어 마에스트로 11기 팀 Fireban 라즈베리파이, 서버 탐지용 

## Server Setup

```
sudo su
sh init.sh
```

### Server Detection Setup

```
pip3 install -r requirements.txt
// psasn1-modules 관련 오류 발생시 "sudo apt-get remove python3-pyasn1-modules"


// "redis start"
redis-server &

// "app.py start"
nohup python3 -u app.py & > app.log &

// "celery start"
nohup celery -A app.celery worker --loglevel=info & > celery.log &
```

### Server Train run

```
sudo su
sh train_sh.sh
```

### Server Auto Train Setup

" * " 부분에 원하는 시간 지정.
chmod 755 /home/fireban/detection/train/train_sh.sh

sudo crontab -e
```
* * * * * root /home/fireban/detection/train/train_sh.sh >> /home/fireban/detection/train/train_sh.log 2>&1
```

## Raspberrypi Setup

```
cd ./detection/raspberry-code

sudo su
sh rasp_init.sh
```

### Raspberrypi Detection Setup

```
python3 -m pip install /tmp/PyTorch-and-Vision-for-Raspberry-Pi-4B/torch-1.4.0a0+f43194e-cp37-cp37m-linux_armv7l.whl
python3 -m pip install /tmp/PyTorch-and-Vision-for-Raspberry-Pi-4B/torchvision-0.5.0a0+9cdc814-cp37-cp37m-linux_armv7l.whl
pip3 install -r requirements.txt
```

## Raspberrypi Detection Auto Update

아래 경로의 스크립트 제일 마지막에 해당 코드를 넣고 저장해줍니다.

sudo vim /etc/profile.d/bash_completion.sh

```
cd /home/pi/detection

ONLINE=1
while [ $ONLINE -ne 0 ]
do
   ping -q -c 1 -w 1 www.github.com >/dev/null 2>&1
   ONLINE=$?
   if [ $ONLINE -ne 0 ]
     then
       sleep 5
   fi
done
echo "We are on line!"

sudo git pull origin master

cd raspberry-code
sudo python3 detect.py &

cd /home/pi
```


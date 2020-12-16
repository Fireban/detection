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

## Raspberrypi Setup

```
apt-get update
apt-get upgrade
apt-get install git

git clone "THIS REPO"
cd SWmaestro-fireban-detection/raspberry-code

sudo su
sh rasp_init.sh
```

### Raspberrypi Detection Setup

```
python3 -m pip install tmp/PyTorch-and-Vision-for-Raspberry-Pi-4B/torch-1.4.0a0+f43194e-cp37-cp37m-linux_armv7l.whl
python3 -m pip install tmp/PyTorch-and-Vision-for-Raspberry-Pi-4B/torchvision-0.5.0a0+9cdc814-cp37-cp37m-linux_armv7l.whl
pip3 install -r requirements.txt
```


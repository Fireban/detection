cd /home/fireban/server_ssh/detection/train

python3 first_data_generate.py
python3 update_data_generate.py

python3 train.py --type=yolo
python3 train.py --type=yolotiny

git add ../raspberry-code/weights/weights_for_run.pth
git commit -m 'model update'
git push origin master

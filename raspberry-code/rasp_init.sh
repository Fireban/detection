echo Y | apt-get update
echo Y | apt-get upgrade

cd /tmp
git clone https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B.git

echo Y | apt-get install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
echo Y | apt-get install libavutil-dev libavcodec-dev libavformat-dev libswscale-dev
echo Y | apt-get install libatlas3-base python3-pip
echo "install dependency"
apt-get update
echo Y | apt-get upgrade
apt-get install -y python3
apt-get install -y python3-pip
echo Y | apt-get install python3-tk
echo Y | apt-get install libgl1-mesa-glx
echo Y | apt-get install redis-server
python3 -m pip install --upgrade pip

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install
git lfs pull
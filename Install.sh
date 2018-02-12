#!/bin/bash

# update apt-get
 apt-get update
# Install apt-get deps
 sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev  

#install nvidia drivers
curl -O "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb"
dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
apt-get update
#  Added this to make sure we don't drag down the newest version of cuda!
apt-get install cuda-9-0 -y

check nvidia driver install
nvidia-smi   

#Install cudnn
CUDNN_DOWNLOAD_SUM=1a3e076447d5b9860c73d9bebe7087ffcb7b0c8814fd1e506096435a2ad9ab0e
curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/cudnn-9.0-linux-x64-v7.tgz -O
echo "$CUDNN_DOWNLOAD_SUM cudnn-9.0-linux-x64-v7.tgz" | sha256sum -c - 
sudo tar -xzvf  cudnn-9.0-linux-x64-v7.tgz 
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

source ~/.bashrc

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh   

# press s to skip terms   

# Do you approve the license terms? [yes|no]
# yes

# Miniconda3 will now be installed into this location:
# accept the location

# Do you wish the installer to prepend the Miniconda3 install location
# to PATH in your /home/ghost/.bashrc ? [yes|no]
# yes    

source ~/.bashrc

#Create conda env to install tf
conda create -n tensorflow
# Activate env
source activate tensorflow   

# Install tensorflow with GPU support for python 3.6
pip3 install tensorflow-gpu==1.5
# If the above fails, try the part below
# pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0-cp36-cp36m-linux_x86_64.whl
 
# run test script   
  echo "import tensorflow as tf   

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))" | python3


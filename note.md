# Getting started

* Create a /checkpoint directory in the /vatt directory and download the pretrained base-base-small model from <https://github.com/google-research/google-research/tree/master/vatt>
* Confirmed to be working with python 3.9.16
* updated the requirements.txt so that it works with latest version
* also do not expand the word embeddings into ./vatt but instead expand in root directory

## Dataset

* Look into ./vatt/data/datasets directory
  * We need to create our own version of the toy_dataset.py for our data
  * we need to store our version of the tfrecord(details in the DMVR website) and give the model we create a path to our tf records
  * VATT website: <https://github.com/google-research/google-research/tree/master/vatt>
  * DMVR website: <https://github.com/deepmind/dmvr/tree/master/examples>

## Debugging

* undefined symbol: cudaGraphInstantiateWithFlags, version libcudart.so.11.0
  * try updating cuda and everything
  * check path of the file by finding: sudo find / -name 'libcudart.so.11.0'
  * check if the file is in the path: echo $LD_LIBRARY_PATH
  * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/targets/x86_64-linux/lib
  * Checking installed cuda version: nvidia-smi
  * Checking toolkit version : nvcc --version
* loading the weights into the model
  * with the current initialziation, we are loading the modality specific, medium-base-small model
  * also need to have the index file in the checkpoint directory

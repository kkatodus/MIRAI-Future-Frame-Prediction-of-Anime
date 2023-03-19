# Getting started

* Create a /checkpoint directory in the /vatt directory and download the pretrained base-base-small model from <https://github.com/google-research/google-research/tree/master/vatt>
* Confirmed to be working with python 3.9.16
* updated the requirements.txt so that it works with latest version

## Dataset

* Look into ./vatt/data/datasets directory
  * We need to create our own version of the toy_dataset.py for our data
  * we need to store our version of the tfrecord(details in the DMVR website) and give the model we create a path to our tf records
  * VATT website: <https://github.com/google-research/google-research/tree/master/vatt>
  * DMVR website: <https://github.com/deepmind/dmvr/tree/master/examples>

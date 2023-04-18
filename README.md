# MIRAI: Future Frame Prediction of Anime from Previous Frames and Audio Input

This project is a part of ECE324 at the University of Toronto

## Motivation
This project involves using a machine learning model to predict the next anime frame from the previous frames and the audio of the anime. The idea is inspired by the issue in the anime industry that all of the frames need to be drawn by hand, which can make the process of creating anime expensive. In addition, some producers have used computer generated imagery to speed up the process but the viewers are not satisfied with it. 

This repo is  for the VATT + LSTM model

## Architecture
![Architecture](https://cdn.discordapp.com/attachments/1068309893171384330/1097281194820907038/Blank_diagram.png)
The intended architecture uses VATT encoder that takes in multiple modes of input. We are passing in image and audio of the video to the encoder to get the code and the convLSTM decoder generates a prediction for the next few freames. 

## Data Collection
The functions for converting data is under the folder `data`
The main work for data collection is done inside this folder.
We have created these scripts for the data collection pipeline:
* `generate_timestamps.py` generates the timestamps of the beginning of a cut that lasts longer than 20 frames (we discard any cut that is smaller than 20 frames so that the cuts are meaningful)
* `split_video.py` split the video into cuts with 20 frames images and save them
* `generate_npy_from_jpeg.py` converts the jpeg files into a single npy file and also resize the image to the desired size. For the project we convert the image into 64x64 pixels.
*  `visualize.py` converts the numpy array back into image and videos for visualizing the results

General Data Collection Pipeline:
![Data Collection Pipeline](https://cdn.discordapp.com/attachments/1068310042908041297/1096509083785383946/data_processing.png)



## VATT
The code for running VATT encoder is under the folder `VATT`
VATT model is configured to the configuration that runs with out input. The configuration is set in `main.py` inside the root folder. The dataloader that VATT required does not work with our dataset, hence needs to overwrite through the configurations.

## LSTM Decoder
A 2D convLSTM is defined in `main.py`
The output from the VATT encoder is passed into the first hidden state of the decoder.
# MIRAI: Future Frame Prediction of Anime from Previous Frames and Audio Input

This project is a part of ECE324 at the University of Toronto

## Motivation
This project involves using a machine learning model to predict the next anime frame from the previous frames and the audio of the anime. The idea is inspired by the issue in the anime industry that all of the frames need to be drawn by hand, which can make the process of creating anime expensive. In addition, some producers have used computer generated imagery to speed up the process but the viewers are not satisfied with it. 

## Architecture
![Architecture](https://cdn.discordapp.com/attachments/1068310123824550019/1089287912274808882/Arch20Diagram.png)
The intended architecture uses VATT encoder that takes in multiple modes of input. We are passing in image and audio of the video to the encoder to get the code and the convLSTM decoder generates a prediction for the next few freames. 

## Data Collection
The functions for converting data is under the folder `data`
The main work for data collection is done inside this folder.
We have created the three scripts for the data collection pipeline:
* `generate_timestamps.py` generates the timestamps of the beginning of a cut that lasts longer than 20 frames (we discard any cut that is smaller than 20 frames so that the cuts are meaningful)
* `split_video.py` split the video into cuts with 20 frames images and save them
* `generate_npy_from_jpeg.py` converts the jpeg files into a single npy file and also resize the image to the desired size

## Data Visualization
Data visualization involves two functions that convert the numpy arrays back into video cuts. The functions are inside the file `/data_proc/data_proc.py`

General Data Collection Pipeline:
![Data Collection Pipeline](https://cdn.discordapp.com/attachments/1068310123824550019/1089300868714741841/data_processing.jpeg)

## VATT
The code for running VATT encoder is under the folder `VATT`
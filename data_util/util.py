"""
Utility functions for converting data from img and text to numpy arrays and text files
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def jpg_to_np(path):
    """
    This function converts the jpg images to numpy arrays
    :param path: the path to the image
    :return: the numpy array of the image
    """
    img = Image.open(path)
    img = np.array(img)
    return img


def get_one_cut(path):
    """
    This function gets one cut of the video and return a list of numpy array and the text file
    :param path: the path to the folder containing a jpg image and a text file
    :return: the numpy array of the video and the text file
    """
    folder = os.listdir(path)
    list_of_frames = []
    text = ""
    # iterate through all files in the folder
    for file in folder:
        if file.endswith(".jpg"):
            img = jpg_to_np(path + "/" + file)
            list_of_frames.append(img)
        elif file.endswith(".txt"):
            # found a text file
            text = open(path + "/" + file, "r")
            text = text.read()

    return list_of_frames, text

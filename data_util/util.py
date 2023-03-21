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


<< << << < HEAD: data/util.py


def save_to_file(path, list_of_frames):
    """ This function saves the list_of_frames to a file as 4D numpy array
    """
    array = np.array(list_of_frames)
    np.save(path, array)
    return array


def check_frame_difference(first_frame, second_frame, threshold=20):
    """ this function checks whether the difference between the two frames is greater than the threshold
    """
    # compute the absolute difference between the two frames
    diff = cv2.absdiff(first_frame, second_frame)

    # convert the difference to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # create a binary image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return np.sum(thresh > 0)


if __name__ == "__main__":
    path = "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/videos/out_of_sight_1"
    list_of_frames, text = get_one_cut(path)
    print(len(list_of_frames))

    array = save_to_file(
        "/Users/punyaphatsuk/Documents/GitHub/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/test.npy", list_of_frames)
    print(array.shape)

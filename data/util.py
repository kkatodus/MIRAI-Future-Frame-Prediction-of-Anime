"""
Utility functions for converting data from img and text to numpy arrays and text files
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.metrics import structural_similarity as compare_ssim


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


def save_to_file(path, list_of_frames):
    """ This function saves the list_of_frames to a file as 4D numpy array
    """
    array = np.array(list_of_frames)
    np.save(path, array)
    return array


def check_frame_difference(first_frame, second_frame, threshold=0.65):
    """ this function checks whether the difference between the two frames is greater than the threshold
    """
    """
    # compute the absolute difference between the two frames
    diff = cv2.absdiff(first_frame, second_frame)
    # print("diff: ", diff)

    # convert the difference to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # create a binary image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    """

    # this part uses SSIM to compare the two frames
    gray1 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
    ssim = compare_ssim(gray1, gray2)
    # print("ssim", ssim)
    return ssim < threshold


def get_cut_timestamps(video_path, threshold=0.65):

    # load video
    cap = cv2.VideoCapture(video_path)

    # check if it has loaded
    if not cap.isOpened():
        print("Error opening video stream or file")

    # get fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # get first frame
    ret, first_frame = cap.read()

    # initialize list of timestamps
    timestamps = []
    n = 0
    # iterate through all frames
    while ret:
        n += 1
        ret, frame = cap.read()
        cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if ret:
            # print("frame number: ", n)
            # print("cur_time: ", cur_time)
            if check_frame_difference(first_frame, frame, threshold) > 0:
                # if the difference is greater than the threshold, add the timestamp to the list
                cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                timestamps.append(cur_time)

            first_frame = frame
        else:
            print("Break from the while loop at cur_time: ", cur_time)
            break

    cap.release()
    return timestamps


if __name__ == "__main__":
    threshold = 0.67
    print("Threshold: ", threshold)
    path = "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/videos/out_of_sight_1"
    list_of_frames, text = get_one_cut(path)
    print(len(list_of_frames))
    timestamps = get_cut_timestamps(
        "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/out_of_sight.mp4", threshold)
    print("timestamps: ", timestamps)
    print("len(timestamps): ", len(timestamps))
    """
    array = save_to_file(
        "/Users/punyaphatsuk/Documents/GitHub/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/out_of_sight_1.npy", list_of_frames)
    print(array.shape)
    """

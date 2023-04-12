import cv2
import numpy as np
from data_proc.data_procviser import visualize_np_sequence_opencv, output_images_from_np_sequence

video_data_np = np.load('/Volumes/OneTouch/Avatar/numpy/1_3.npy')
video_shape = video_data_np.shape
print("video_data_np", video_data_np.shape)
visualize_np_sequence_opencv(video_data_np[14], video_name="video.mp4", fps=10)

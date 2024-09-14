import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from common.camera import *
from common.model import *
from common.utils import Timer, evaluate
from common.generators import UnchunkedGenerator
from bvh_skeleton import h36m_skeleton
import warnings
warnings.filterwarnings("error")

# Record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()

time0 = ckpt_time()

# Function to capture and process video from camera
def get_video_points_data():
    # Open camera
    cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    all_points = np.zeros((1, 17, 3))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, stream=True, device='cpu')
        for result in results:
            keypoints = result.keypoints
            res_plotted = result.plot(boxes=False)
            cv2.imshow("result", res_plotted)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if keypoints.data.shape[1] == 17:
                all_points = np.concatenate((all_points, keypoints.data), axis=0)
            else:
                print(17, keypoints.data.size(), keypoints.data.shape, keypoints.data)
    cap.release()
    cv2.destroyAllWindows()
    return all_points

# Convert 2D points to 3D
def get_3d_points(points_2d):
    # Similar processing as the original script
    # This part should include normalization, model loading, and prediction
    # This is just a placeholder as implementation details depend on specific libraries and model architecture used
    pass

# Main function
if __name__ == "__main__":
    # Load the YOLOv8 model
    model = YOLO('../yolov8n-pose.pt')
    metadata = {'layout_name': 'coco', 'num_joints': 17,
                'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

    all_points = get_video_points_data()
    get_3d_points(all_points)

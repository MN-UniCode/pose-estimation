import os
import sys

import cv2

# Mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from classes.body_landmarks import BodyLandmarks

# Filters
from classes.filters import ButterworthMultichannel, HampelMultichannel

# Utilities and data structures
from classes.kinetix import Kinetix


# Kinetic energy computation
use_anthropometric_tables = True
total_mass = 67
# frame_width = 1920
# frame_height = 1080

# Paths and files
base_path = ""
video_path = "project/videos/"
model_path = "project/models/"
video_name = "mauri.mp4"
live_input = False

# Filtering
cutoff = 3.0
order = 2

# Plotting
plot_window_seconds = 5
max_ke = 12.0

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Creating a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path=base_path + model_path + 'pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# === Pre-processing === #

# Selecting the input source (either a file or a video camera)
if not live_input:
    path = base_path + video_path + video_name
    cap = cv2.VideoCapture(path)
    print(f"Processing file: {path}.")
else:
    cap = cv2.VideoCapture(0)
    print("Processing webcam input.")

fps = cap.get(cv2.CAP_PROP_FPS)
while True:
    result = input(f"Is your video at {fps} fps (y/n)\n")
    if result == "y":
        break
    else:
        sys.exit()


# Creating filter
num_channels = len(BodyLandmarks) * 3

hampel_filter = HampelMultichannel(num_channels, window_size=11, n_sigma=2.5, replace_with='median')

butterworth_filter = ButterworthMultichannel(num_channels, order, cutoff, btype='lowpass', fs=fps)

filters = [butterworth_filter]

kinetix = Kinetix(fps, plot_window_seconds, total_mass)

kinetix(detector, filters, cap, max_ke, use_anthropometric_tables)
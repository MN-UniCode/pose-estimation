import cv2

# Mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from classes.body_landmarks import BodyLandmarks

# Filters
from classes.filters import ButterworthMultichannel

# Utilities and data structures
from classes.kinetix import Kinetix


# Kinetic energy computation
use_anthropometric_tables = True
total_mass = 67
frame_width = 1920
frame_height = 1080

# Paths and files
base_path = ""
video_path = "03_low_level_features/videos/"
model_path = "project/models/"
video_name = "micro-dance.avi"
live_input = False

# Filtering
fps = 25
cutoff = 3.0
order = 2

# Plotting
plot_window_seconds = 5
max_ke = 15.0


# Creating a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path=base_path + model_path + 'pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# === Pre-processing === #
# Creating filter
butterworth_filter = ButterworthMultichannel(len(BodyLandmarks) * 3, order, cutoff, btype='lowpass', fs=fps)

# Selecting the input source (either a file or a video camera)
if not live_input:
    path = base_path + video_path + video_name
    cap = cv2.VideoCapture(path)
    print(f"Processing file: {path}.")
else:
    cap = cv2.VideoCapture(0)
    print("Processing webcam input.")

kinetix = Kinetix(fps, plot_window_seconds, frame_width, frame_height, total_mass)

kinetix(detector, butterworth_filter, cap, max_ke, use_anthropometric_tables)


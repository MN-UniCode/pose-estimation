# Import libraries

# OpenCV
import cv2

# Numpy
import numpy as np

# Mediapipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Matplotlib
import matplotlib.pyplot as plt

# Constants
base_path = ""
video_path = "02_motion_tracking/videos/"
model_path = "02_motion_tracking/models/"
video_name = "micro-dance.avi"
live_input = True

# Defining a function to draw body landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = rgb_image.copy()
    
    # Loop through the detected poses to visualize
    if len(pose_landmarks_list) > 0:
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            
            # Draw the pose landmarks
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

# Creating a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path=base_path + model_path + 'pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Selecting the input source (either a file or a video camera)
if not live_input:
    path = base_path + video_path + video_name
    cap = cv2.VideoCapture(path)
    print(f"Processing file: {path}.")
else:
    cap = cv2.VideoCapture(0)
    print("Processing webcam input.")

# Checking for possible errors
if not cap.isOpened():
    print("Error in opening the video stream.")
    sys.exit()
cap.set(3,640) # adjust width
cap.set(4,480) # adjust height

while True:
    
    # Getting current frame
    success, current_frame = cap.read()
    
    # For each frame, detect landmarks, draw, and display them
    if success:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=current_frame)
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(current_frame, detection_result)
        cv2.imshow("Mediapipe", annotated_image)
        if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
            cap.release()
            break
    else:
        break

# Closing video capture device
cv2.destroyAllWindows() 
cv2.waitKey(1)
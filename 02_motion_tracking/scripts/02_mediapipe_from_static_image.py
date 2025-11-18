# IMPORTING LIBRARIES
import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt

model_path = '02_motion_tracking/models/'
image_path = '02_motion_tracking/videos/'

# DEFINING A FUNCTION TO DRAW BODY LANDMARKS
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  if annotated_image.shape[2] == 4:
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGBA2RGB)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
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

# STEP 1: Creating a PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path= model_path + 'pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 2: Loading the input image.
image = mp.Image.create_from_file(image_path + 'dancer.jpg')

# STEP 3: Detecting pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 4: Processing the detection result. In this case, just visualizing it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# set size
plt.figure(figsize=(10,10))
plt.axis("off")

# convert color from CV2 BGR back to RGB
image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
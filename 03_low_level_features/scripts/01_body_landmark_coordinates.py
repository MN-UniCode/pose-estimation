# === Import libraries === #

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
from body_landmarks import BodyLandmarks

# Filters
from filters import Butterworth, SavitzkyGolay, Hampel

# Utilities and data structures
import time
import sys
from collections import deque


# === Constants  === #

# Paths and files
base_path = ""
video_path = "02_motion_tracking/videos/"
model_path = "02_motion_tracking/models/"
video_name = "micro-dance.avi"
live_input = False

# Selected landmark
body_landmark_index = BodyLandmarks.NOSE
bottom_up_origin = False
frame_width = 352
frame_height = 288

# Filtering
apply_filtering = True
remove_outliers = True
fps = 25          # Set your actual frame rate
cutoff = 3.0      # Hz - smooth but responsive
order = 2

# Plotting
plot_window_seconds = 5
min_value_x = 0.0  # Adjust based on expected max value of x
max_value_x = 1.0  # Adjust based on expected min value of x
min_value_y = 0.0  # Adjust based on expected max value of y
max_value_y = 1.0  # Adjust based on expected min value of y
min_value_z = -0.5  # Adjust based on expected max value of z
max_value_z = 0.5 # Adjust based on expected min value of z


# === Features === #

# Getting a body landmark's coordinates
def get_landmark_coords(detection_result, body_landmark_index, bottom_up_origin=False):
    if not detection_result:
        return None, None, None
    
    if not detection_result.pose_landmarks:
        return None, None, None

    landmarks = detection_result.pose_landmarks[0]  # First person

    if body_landmark_index >= len(landmarks):
        print(f"Invalid index: {body_landmark_index}")
        return None, None, None

    selected_landmark = landmarks[body_landmark_index]

    if bottom_up_origin:
        y = 1 - selected_landmark.y
    else:
        y = selected_landmark.y

    coordinates = np.array([
        selected_landmark.x,
        y,
        selected_landmark.z
    ])

    visibility = selected_landmark.visibility
    presence = selected_landmark.presence

    return coordinates, visibility, presence


# === Auxiliary functions === #

# Drawing a single body landmark
def draw_single_landmark_on_image(rgb_image, detection_result, body_landmark_index):
    annotated_image = rgb_image.copy()
    coord, visibility, _ = get_landmark_coords(detection_result, body_landmark_index)
    if coord is not None:

        # Create a new NormalizedLandmarkList with just one landmark
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.append(
            landmark_pb2.NormalizedLandmark(
                x=coord[0],
                y=coord[1],
                z=coord[2],
                visibility = visibility
            )   
        )

        # Draw the single landmark using MediaPipe utils
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=landmark_list,
            connections=[],  # No connections → just the point
            landmark_drawing_spec=solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0),
                thickness=4,
                circle_radius=5
            )
        )

    return annotated_image

 # Drawing the graph of a time-series using OpenCV
def draw_cv_graph(history, width=640, height=240, min_value=0.0, max_value=2.0, fps=25, window_length=5, y_label="y"):
    graph = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

    # Axes
    cv2.line(graph, (50, 0), (50, height - 40), (0, 0, 0), 1)  # y-axis
    cv2.line(graph, (50, height - 40), (width, height - 40), (0, 0, 0), 1)  # x-axis

    # Y-axis labels and grid lines
    for i in range(5):
        y_value = min_value + (max_value - min_value) * i / 4
        y_pos = int(height - 40 - ((y_value - min_value) / (max_value - min_value)) * (height - 50))
        label = f"{y_value:.1f}"
        cv2.putText(graph, label, (5, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        # Optional: draw horizontal grid lines
        cv2.line(graph, (50, y_pos), (width, y_pos), (230, 230, 230), 1)

    # Draw horizontal zero-line if in range
    if min_value < 0 < max_value:
        zero_y = int(height - 40 - ((0 - min_value) / (max_value - min_value)) * (height - 50))
        cv2.line(graph, (50, zero_y), (width, zero_y), (200, 200, 200), 1, lineType=cv2.LINE_AA)

    # X-axis time ticks
    history_length = fps * window_length
    seconds_range = history_length / fps
    tick_px = (width - 50) / seconds_range

    for i in range(int(seconds_range) + 1):
        x = int(50 + i * tick_px)
        cv2.line(graph, (x, height - 40), (x, height - 35), (0, 0, 0), 1)
        cv2.putText(graph, f"{i}s", (x - 10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Plot line
    if len(history) >= 2:
        for i in range(1, len(history)):
            x1 = int(50 + (i - 1) / history_length * (width - 50))
            x2 = int(50 + i / history_length * (width - 50))

            # Clamp values between min and max
            y1_val = max(min(history[i - 1], max_value), min_value)
            y2_val = max(min(history[i], max_value), min_value)

            y1 = int(height - 40 - ((y1_val - min_value) / (max_value - min_value)) * (height - 50))
            y2 = int(height - 40 - ((y2_val - min_value) / (max_value - min_value)) * (height - 50))

            cv2.line(graph, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Axis labels
    cv2.putText(graph, y_label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(graph, "Time (s)", (width // 2, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    return graph

# Stack OpenCV images horizontally
def stack_images_horizontal(images, scale=1.0):
    resized_images = []
    for img in images:
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        resized_images.append(img)
    return cv2.hconcat(resized_images)

# === Main === #
x_history = deque(maxlen=fps*plot_window_seconds)
y_history = deque(maxlen=fps*plot_window_seconds)
z_history = deque(maxlen=fps*plot_window_seconds)

# Creating a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path=base_path + model_path + 'pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Creating filters
butterworh_filter_x = Butterworth(order, cutoff, 'lowpass', fps)
butterworh_filter_y = Butterworth(order, cutoff, 'lowpass', fps)
butterworh_filter_z = Butterworth(order, cutoff, 'lowpass', fps)
hampel_filter_x = Hampel(window_size=12)
hampel_filter_y = Hampel(window_size=12)
hampel_filter_z = Hampel(window_size=12)

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

while True:
    # Getting current frame
    success, current_frame = cap.read()
    if not success:
        break
    
    # Resizing it
    current_frame = cv2.resize(current_frame, (frame_width, frame_height))

    # Running detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=current_frame)
    detection_result = detector.detect(mp_image)

    # Getting current time and coordinates of the selected landmark
    x = 0
    y = 0
    z = 0
    
    curr_coord, _, _ = get_landmark_coords(detection_result, body_landmark_index, bottom_up_origin)

    if curr_coord is not None:
        x = curr_coord[0]
        y = curr_coord[1]
        z = curr_coord[2]

        # If filters have to be applied, doing it
        if apply_filtering:
            filtered_x = butterworh_filter_x.filter(curr_coord[0])
            filtered_y = butterworh_filter_y.filter(curr_coord[1])
            filtered_z = butterworh_filter_z.filter(curr_coord[2])
            x = filtered_x
            y = filtered_y
            z = filtered_z

    # If outliers have to be remove, doing it
    if remove_outliers:
        clean_x = hampel_filter_x.filter(x)
        clean_y = hampel_filter_y.filter(y)
        clean_z = hampel_filter_z.filter(z)
        x = clean_x
        y = clean_y
        z = clean_z

    x_history.append(x)
    y_history.append(y)
    z_history.append(z)

    # Plotting
    annotated_image = draw_single_landmark_on_image(current_frame, detection_result, body_landmark_index)
    x_graph_image = draw_cv_graph(x_history, annotated_image.shape[1], annotated_image.shape[0], min_value_x, max_value_x, fps, plot_window_seconds, "x")
    y_graph_image = draw_cv_graph(y_history, annotated_image.shape[1], annotated_image.shape[0], min_value_y, max_value_y, fps, plot_window_seconds, "y")
    z_graph_image = draw_cv_graph(z_history, annotated_image.shape[1], annotated_image.shape[0], min_value_z, max_value_z, fps, plot_window_seconds, "z")
    combined = stack_images_horizontal([annotated_image, x_graph_image, y_graph_image, z_graph_image])

    # Showing the result
    cv2.imshow("Landmark 3D coordinates", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
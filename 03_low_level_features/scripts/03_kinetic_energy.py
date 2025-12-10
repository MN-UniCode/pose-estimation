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
import masses

# Filters
from filters import ButterworthMultichannel

# Utilities and data structures
import time
import sys
from collections import deque


# === Constants  === #

# Kinetic energy computation
use_anthropometric_tables = True
total_mass = 67
frame_width = 352
frame_height = 288

# Paths and files
base_path = ""
video_path = "project/videos/"
model_path = "02_motion_tracking/models/"
video_name = "nico.MOV"
live_input = False

# Filtering
apply_filtering = True
fps = 25          # Set your actual frame rate
cutoff = 3.0
order = 2

# Plotting
plot_window_seconds = 5
max_ke = 12.0  # Adjust based on expected max kinetic energy


# === Features === #

# Computing the first-order derivative
def first_order_derivative(curr_value, prev_value, curr_time, prev_time):
    result = None
    if curr_value is not None and prev_value is not None and curr_time is not None and prev_time is not None:
        dt = curr_time - prev_time
        result = (curr_value - prev_value) / dt
    return result

def compute_kinetic_energy(current_detection_result, previous_detection_result,
    prev_time, curr_time,
    masses=None,
    apply_filtering=False, velocity_filter=None):

    # Ensure landmarks exist
    if detection_result is None or previous_detection_result is None:
        return None
    
    if previous_detection_result.pose_world_landmarks is None or len(previous_detection_result.pose_world_landmarks) == 0:
        return None
    
    if current_detection_result.pose_world_landmarks is None or len(current_detection_result.pose_world_landmarks) == 0:
        return None

    # Use only the first detected person
    current_landmarks = current_detection_result.pose_world_landmarks[0]
    previous_landmarks = previous_detection_result.pose_world_landmarks[0]

    curr_p = np.array([[lm.x, lm.y, lm.z] for lm in current_landmarks])
    prev_p = np.array([[lm.x, lm.y, lm.z] for lm in previous_landmarks])

    # Filter POSITIONS, not velocities
    if apply_filtering and velocity_filter is not None:
        curr_p = velocity_filter.filter(curr_p.reshape(-1)).reshape(curr_p.shape)
        prev_p = velocity_filter.filter(prev_p.reshape(-1)).reshape(prev_p.shape)

    n_landmarks = len(current_landmarks)
    if masses is None:
        masses = np.ones(n_landmarks)

    # Compute velocity vectors for each landmark
    dt = curr_time - prev_time
    velocities = (curr_p - prev_p) / dt

    # Optional filtering
    # if apply_filtering and velocity_filter is not None:
    #     # Flatten velocities for filtering: (n_points * 3,)
    #     v_flat = velocities.reshape(-1)
    #     # Filter one "sample" per channel (vectorized)
    #     v_f = velocity_filter.filter(v_flat)
    #     velocities = v_f.reshape(n_landmarks, 3)

    # Compute total kinetic energy
    speed_squared = np.sum(velocities**2, axis=1)
    total_ke = 0.5 * np.sum(masses * speed_squared)

    if total_ke > 1:
        print("Total ke = ", total_ke)

    return total_ke
     
   
# === Auxiliary functions === #

# Drawing body landmarks
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

# Drawing a graph over time using OpenCV
def draw_cv_graph(history, width=1240, height=480, max_value = 2.0, fps = 25, window_length = 5, y_label="y"):
    graph = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

    # Axes
    cv2.line(graph, (50, 0), (50, height - 40), (0, 0, 0), 1)  # y-axis
    cv2.line(graph, (50, height - 40), (width, height - 40), (0, 0, 0), 1)  # x-axis

    # Y-axis labels (fixed range)
    for i in range(5):
        y_value = max_value * i / 4
        y_pos = int(height - 40 - (y_value / max_value) * (height - 50))
        label = f"{y_value:.1f}"
        cv2.putText(graph, label, (5, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

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

            y1 = int(height - 40 - (min(history[i - 1], max_value) / max_value) * (height - 50))
            y2 = int(height - 40 - (min(history[i], max_value) / max_value) * (height - 50))

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
prev_detection = None
prev_time = None
curr_time = None
ke_history = deque(maxlen=fps*plot_window_seconds)

# Creating a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path=base_path + model_path + 'pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Creating filter
butterworth_filter = ButterworthMultichannel(len(BodyLandmarks)*3, order, cutoff, btype='lowpass', fs=fps)

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

    frame_height, frame_width, _ = current_frame.shape
    
    # Resizing it
    current_frame = cv2.resize(current_frame, (frame_width, frame_height))

    # Running detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=current_frame)
    detection_result = detector.detect(mp_image)

    # Getting current time and coordinates of the selected landmark
    curr_time = time.time()

    # Computing kinetic energy
    if use_anthropometric_tables:
        masses_vector = masses.create_mass_vector(total_mass)
    else:
        masses_vector = None
    ke = compute_kinetic_energy(detection_result, prev_detection, 
                                prev_time, curr_time, masses_vector,
                                apply_filtering, butterworth_filter)
    if ke is not None:
        ke_history.append(ke)
    else:
        ke_history.append(0.0)

    # Updating
    prev_detection = detection_result
    prev_time = curr_time
        
    # Plotting
    annotated_image = draw_landmarks_on_image(current_frame, detection_result)
    ke_graph_image = draw_cv_graph(ke_history, height=annotated_image.shape[0], max_value=max_ke, fps=fps, window_length=plot_window_seconds, y_label="Kinetic Energy")
    combined = stack_images_horizontal([annotated_image, ke_graph_image])

    # Showing the result
    cv2.imshow("Landmarks overall kinetic energy", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
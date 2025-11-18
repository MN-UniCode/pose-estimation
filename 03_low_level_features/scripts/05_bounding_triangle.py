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
from body_landmarks import BodyLandmarkGroups

# Filters
from filters import Butterworth

# Utilities and data structures
import sys
from collections import deque


# === Constants  === #

# Bounding triangle computation
frame_width = 352
frame_height = 288

# Paths and files
base_path = ""
video_path = "02_motion_tracking/videos/"
model_path = "02_motion_tracking/models/"
video_name = "micro-dance.avi"
live_input = False

# Filtering
apply_filtering = True
fps = 25          # Set your actual frame rate
cutoff = 3.0
order = 2

# Plotting
plot_window_seconds = 5
min_pd = 0.0  # Adjust based on expected min point density
max_pd = 0.1  # Adjust based on expected max point density


# === Features === #

def compute_bounding_triangle(detection_result, use_mid_head=True):

    if detection_result is None:
        return None, None
    
    if detection_result.pose_landmarks is None or len(detection_result.pose_landmarks) == 0:
        return None, None
    
    body_landmarks = detection_result.pose_landmarks[0]
    if not body_landmarks or len(body_landmarks) == 0:
        return None, None
    
    if len(body_landmarks) < 33:
        return None, None

    # MediaPipe Pose indices
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_EYE = 2
    RIGHT_EYE = 5
    NOSE = 0

    # Hands
    left_hand = np.array([
        body_landmarks[LEFT_WRIST].x,
        body_landmarks[LEFT_WRIST].y,
        body_landmarks[LEFT_WRIST].z
    ])
    right_hand = np.array([
        body_landmarks[RIGHT_WRIST].x,
        body_landmarks[RIGHT_WRIST].y,
        body_landmarks[RIGHT_WRIST].z
    ])

    # Head (either midpoint of eyes or nose)
    if use_mid_head:
        left_eye = np.array([
            body_landmarks[LEFT_EYE].x,
            body_landmarks[LEFT_EYE].y,
            body_landmarks[LEFT_EYE].z
        ])
        right_eye = np.array([
            body_landmarks[RIGHT_EYE].x,
            body_landmarks[RIGHT_EYE].y,
            body_landmarks[RIGHT_EYE].z
        ])
        head = (left_eye + right_eye) / 2
    else:
        head = np.array([
            body_landmarks[NOSE].x,
            body_landmarks[NOSE].y,
            body_landmarks[NOSE].z
        ])

    # Return the triangle points (Left hand, Right hand, Head)
    triangle = np.stack([left_hand, right_hand, head])

    # Compute area using 3D cross product formula
    a, b, c = triangle
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    return triangle, area


# === Auxiliary functions === #

# Drawing body landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
    if not detection_result:
        return rgb_image
    
    if not detection_result.pose_landmarks:
        return rgb_image
    
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


# Drawing the bouning triangle
def draw_bounding_triangle(frame, triangle,
    color=(0, 255, 0), thickness=2,
    draw_points=True, fill=False, fill_opacity=0.3
):
    if triangle is None or triangle.shape != (3, 3):
        return frame

    h, w, _ = frame.shape

    # Convert normalized coordinates to pixel coordinates
    pts = np.array([
        [int(triangle[i, 0] * w), int(triangle[i, 1] * h)]
        for i in range(3)
    ], dtype=np.int32).reshape((-1, 1, 2))

    if fill:
        # Create an overlay to handle transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)

        # Blend with original frame
        frame = cv2.addWeighted(overlay, fill_opacity, frame, 1 - fill_opacity, 0)
    else:
        # Draw the triangle outline
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    # Optionally draw vertex points
    if draw_points:
        for (x, y) in pts.reshape(-1, 2):
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    return frame


def draw_cv_graph(history, width=640, height=240, min_value=0.0, max_value=2.0, fps=25, window_length=5, y_label="y"):
    graph = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

    # Axes
    cv2.line(graph, (50, 0), (50, height - 40), (0, 0, 0), 1)  # y-axis
    cv2.line(graph, (50, height - 40), (width, height - 40), (0, 0, 0), 1)  # x-axis

    # Y-axis labels and grid lines
    for i in range(5):
        y_value = min_value + (max_value - min_value) * i / 4
        y_pos = int(height - 40 - ((y_value - min_value) / (max_value - min_value)) * (height - 50))
        label = f"{y_value:.2f}"
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
history = deque(maxlen=fps*plot_window_seconds)

# Creating a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path=base_path + model_path + 'pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Creating filter
butterworth_filter = Butterworth(order, cutoff, 'lowpass', fps)

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

    # Computing kinetic energy
    triangle, area = compute_bounding_triangle(detection_result)

    if area is not None:
        if apply_filtering:
            filtered_area = butterworth_filter.filter(area)
            history.append(filtered_area)
        else:
            history.append(area)
    else:
       history.append(0.0)

    # Plotting
    annotated_image_1 = draw_landmarks_on_image(current_frame, detection_result)
    annotated_image_2 = draw_bounding_triangle(annotated_image_1, triangle, fill=True, fill_opacity=0.9)
    pd_graph_image = draw_cv_graph(history, annotated_image_2.shape[1], annotated_image_2.shape[0], min_pd, max_pd, fps, plot_window_seconds, "Bounding triangle area")
    combined = stack_images_horizontal([annotated_image_2, pd_graph_image])

    # Showing the result
    cv2.imshow("Bounding triangle", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
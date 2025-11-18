import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import sys

# Global variables
centroid_trail_length = 30
centroid_trail = deque(maxlen=centroid_trail_length)

# Auxiliary functions
def nothing(x):
    pass

def resize_centroid_trail(x):
    global centroid_trail
    global centroid_trail_length
    centroid_trail_length = cv2.getTrackbarPos("Centroid trail length", "YOLOv8 Person Tracking")
    centroid_trail.clear()
    centroid_trail = deque(maxlen=centroid_trail_length)
    
def get_centroid(mask, x1, y1, x2, y2):
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)
    else:
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)  # fallback to box center
    return centroid

# Constants
base_path = ""
video_path = "02_motion_tracking/videos/"
video_name = "micro-dance.avi"
live_input = False

model = "02_motion_tracking/models/yolov8n-seg.pt"
person_class_id = 0
confidence_threshold = 0.3

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

# Creating the interface for the user
print("Press 'r' to reset tracked person. Press 'q' to quit the program.")
cv2.namedWindow("YOLOv8 Person Tracking")
cv2.createTrackbar("Confidence threshold", "YOLOv8 Person Tracking", int(confidence_threshold * 100), 100, nothing)
cv2.createTrackbar("Centroid trail length", "YOLOv8 Person Tracking", centroid_trail_length, 100, resize_centroid_trail)

# Load model
model = YOLO(model)
tracked_id = None

while True:

    # Getting the current frame
    ret, frame = cap.read()
    if not ret:
        print("End of the video or error in reading a frame.")
        break

    # Run YOLO with tracking enabled
    conf_th = cv2.getTrackbarPos("Confidence threshold", "YOLOv8 Person Tracking")
    results = model.track(source=frame, persist=True, conf=conf_th / 100.0, verbose=False)[0]

    # Filter person class only
    detections = []
    results = model.track(source=frame, persist=True, conf=conf_th / 100.0, verbose=False)[0]

    # Filter person class only
    detections = []
    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy().astype(np.uint8)
        boxes = results.boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id
        confs = boxes.conf.cpu().numpy()
        
        for i, (cls, conf) in enumerate(zip(classes, confs)):
            if cls == person_class_id:
                if ids[i] is not None:
                    track_id = int(ids[i].item()) if ids is not None else -1
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                    mask = masks[i]
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)
                    centroid = get_centroid(mask_resized , x1, y1, x2, y2)
                    detections.append({
                        'track_id': track_id,
                        'bbox': (x1, y1, x2, y2),
                        'conf': float(conf),
                        'centroid': centroid
                    })

    # If we don't have a person selected yet, pick the one with the lowest track_id
    if tracked_id is None and detections:
        tracked_id = min(d['track_id'] for d in detections)

    # Find the currently tracked person
    current_person = None
    for d in detections:
        if d['track_id'] == tracked_id:
            current_person = d
            break

    if current_person:
        x1, y1, x2, y2 = current_person['bbox']
        cx, cy = current_person['centroid']
        conf = current_person['conf']

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw centroid
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # Draw labels
        cv2.putText(frame, f"Person ID {tracked_id} Conf {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Centroid: ({cx},{cy})", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Update trail
        centroid_trail.appendleft((cx, cy))

        # Draw trail
        for i in range(1, len(centroid_trail)):
            pt1 = centroid_trail[i - 1]
            pt2 = centroid_trail[i]
            cv2.line(frame, pt1, pt2, (0, 150, 255), 2)

    # Display
    cv2.imshow("YOLOv8 Person Tracking", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        print("Reset tracking.")
        tracked_id = None
        centroid_trail.clear()

cap.release()
cv2.destroyAllWindows()
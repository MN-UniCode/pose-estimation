import cv2
import sys
import numpy as np

# Computing background subtraction
def frame_differencing(prev_frame, curr_frame, threshold_value=30):
    
    # Conversion from color to grayscale
    if len(prev_frame.shape) == 3:
        gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if len(curr_frame.shape) == 3:
        gray_curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Computing the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(gray_prev_frame, gray_curr_frame)
    
    # Applying the threshold to the difference image
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    return diff, thresh

# Auxiliary functions
def nothing(x):
    pass

def stack_images_horizontal(images, scale=1.0):
    resized_images = []
    for img in images:
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        resized_images.append(img)
    return cv2.hconcat(resized_images)

def stack_images_vertical(images, scale=1.0):
    resized_images = []
    for img in images:
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        resized_images.append(img)
    return cv2.vconcat(resized_images)

# Constants
base_path = ""
video_path = "01_background_segmentation/videos/"
video_name = "micro-dance.avi"
live_input = False

threshold_value = 30

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
print("Press 'q' to quit the program.")
cv2.namedWindow("Frame differencing")
cv2.createTrackbar("Threshold", "Frame differencing", threshold_value, 255, nothing)

# Initializing previous frame to a black image
gray_blank = np.zeros((480, 640), dtype=np.uint8)
prev_frame = cv2.cvtColor(gray_blank, cv2.COLOR_GRAY2BGR)

while True:

    # Getting the current frame
    ret, curr_frame = cap.read()
    if not ret:
        print("End of the video or error in reading a frame.")
        break

    # Resizing them
    curr_frame = cv2.resize(curr_frame, (640, 480))

    # Getting the value the user set for the threshold and applying frame differencing
    threshold_value = cv2.getTrackbarPos("Threshold", "Frame differencing")
    diff, thresh = frame_differencing(prev_frame, curr_frame, threshold_value)

    # Updating the previous frame
    prev_frame = curr_frame.copy()

    # Preparing visualization
    top_row = stack_images_horizontal([prev_frame, curr_frame], scale=0.5)
    bottom_row = stack_images_horizontal([diff, thresh], scale=0.5)
    dashboard_img = stack_images_vertical([top_row, bottom_row])

    # Showing results
    cv2.imshow("Frame differencing", dashboard_img)

    # Checking input from the user to leave the application
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        print("Quit the program.")
        break

# Closing everything
cap.release()
cv2.destroyAllWindows()
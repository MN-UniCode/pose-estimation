import cv2
import sys
import numpy as np

# Computing background subtraction
def background_subtraction(current_frame, background, threshold_value=30):
    
    # Conversion from color to grayscale
    if len(current_frame.shape) == 3:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    if len(background.shape) == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # Computing the absolute difference between the current frame and the background
    diff = cv2.absdiff(background, current_frame)
    
    # Applying the threshold to the difference image
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    return current_frame, diff, thresh

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
live_input = True

background = None
first_frame_set = False
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
print("Press 's' to update the background. Press 'q' to quit the program.")
cv2.namedWindow("Simple background subtraction")
cv2.createTrackbar("Threshold", "Simple background subtraction", threshold_value, 255, nothing)

while True:

    # Getting the current frame
    ret, frame = cap.read()
    if not ret:
        print("End of the video or error in reading a frame.")
        break

    # Resizing it
    frame = cv2.resize(frame, (640, 480))

    # If this is the first frame, saving it as the background
    if not first_frame_set:
        background = frame
        first_frame_set = True
        print("First frame taken as the initial background.")

    # Getting the value the user set for the threshold and applying background subtraction
    threshold_value = cv2.getTrackbarPos("Threshold", "Simple background subtraction")
    gray_frame, diff, thresh = background_subtraction(frame, background, threshold_value)

    # Preparing visualization
    top_row = stack_images_horizontal([frame, background], scale=0.5)
    bottom_row = stack_images_horizontal([diff, thresh], scale=0.5)
    dashboard_img = stack_images_vertical([top_row, bottom_row])

    # Showing results
    cv2.imshow("Simple background subtraction", dashboard_img)

    # Checking input from the user to either update the background or leave the application
    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Background manually updated.")
    elif key == ord('q'):
        print("Quit the program.")
        break

# Closing everything
cap.release()
cv2.destroyAllWindows()
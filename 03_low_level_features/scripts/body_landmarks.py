# pose_landmarks.py

from enum import IntEnum, auto

class BodyLandmarks(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY_1 = 17
    RIGHT_PINKY_1 = 18
    LEFT_INDEX_1 = 19
    RIGHT_INDEX_1 = 20
    LEFT_THUMB_2 = 21
    RIGHT_THUMB_2 = 22
    
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    def __str__(self):
        return self.name


class BodyLandmarkGroups:
    """Convenient landmark index groups for MediaPipe Pose."""

    # Full body (default 33 landmarks)
    ALL = list(range(33))

    # Upper body: shoulders, elbows, wrists, hips
    UPPER_BODY = [11, 12, 13, 14, 15, 16, 23, 24]

    # Lower body: hips, knees, ankles
    LOWER_BODY = [23, 24, 25, 26, 27, 28]

    # Arms only
    ARMS = [11, 12, 13, 14, 15, 16]

    # Legs only
    LEGS = [23, 24, 25, 26, 27, 28]

    # Torso (neck/shoulder/hip area)
    TORSO = [11, 12, 23, 24]

    # Head and face (nose, eyes, ears, mouth)
    HEAD = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __str__(self):
        return self.name
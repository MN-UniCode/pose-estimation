from enum import IntEnum

# MediaPipe 33-landmark enumeration
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


# MediaPipe landmark groups
class BodyLandmarkGroups:
    ALL = list(range(33))  # Full body

    UPPER_BODY = [11, 12, 13, 14, 15, 16, 23, 24]  # Shoulders, elbows, wrists, hips
    LOWER_BODY = [23, 24, 25, 26, 27, 28]          # Hips, knees, ankles

    RIGHT_ARM = [12, 14, 16, 18, 20, 22]           # Right arm (shoulder → hand)
    LEFT_ARM = [11, 13, 15, 17, 19, 21]            # Left arm

    RIGHT_LEG = [24, 26, 28]                       # Right leg (hip → ankle)
    LEFT_LEG = [23, 25, 27]                        # Left leg

    TORSO = [11, 12, 23, 24]                       # Shoulder & hip joints
    HEAD = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]      # Head & face landmarks

    def __str__(self):
        return self.name


# YOLO 17-landmark enumeration
class YoloBodyLandmarks(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __str__(self):
        return self.name


# YOLO landmark groups
class YoloBodyLandmarkGroups:
    ALL = list(range(17))  # Full body

    # Upper body: shoulders, elbows, wrists, hips
    UPPER_BODY = [5, 6, 7, 8, 9, 10, 11, 12]

    # Lower body: hips, knees, ankles
    LOWER_BODY = [11, 12, 13, 14, 15, 16]

    # Arms
    RIGHT_ARM = [6, 8, 10]
    LEFT_ARM = [5, 7, 9]

    # Legs
    RIGHT_LEG = [12, 14, 16]
    LEFT_LEG = [11, 13, 15]

    # Torso: shoulders and hips
    TORSO = [5, 6, 11, 12]

    # Head: nose, eyes, ears
    HEAD = [0, 1, 2, 3, 4]

    def __str__(self):
        return self.name

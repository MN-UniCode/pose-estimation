import numpy as np
from classes.body_landmarks import BodyLandmarkGroups, YoloBodyLandmarkGroups

# Anthropometric mass fractions for body segments (% of total body mass)
MASS_FRACTIONS = {
    'head': 0.0694,
    'torso': 0.43,
    'left_upper_arm': 0.0271,
    'right_upper_arm': 0.0271,
    'left_lower_arm': 0.0162,
    'right_lower_arm': 0.0162,
    'left_upper_leg': 0.105,
    'right_upper_leg': 0.105,
    'left_lower_leg': 0.0465,
    'right_lower_leg': 0.0465
}

# Landmark indices per segment for MediaPipe Pose
LANDMARK_GROUPS = {
    'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'torso': [11, 12, 23, 24],
    'left_upper_arm': [11, 13, 15],
    'right_upper_arm': [12, 14, 16],
    'left_lower_arm': [13, 15, 17, 19, 21],
    'right_lower_arm': [14, 16, 18, 20, 22],
    'left_upper_leg': [23, 25, 27],
    'right_upper_leg': [24, 26, 28],
    'left_lower_leg': [25, 27, 29, 31],
    'right_lower_leg': [26, 28, 30, 32]
}

# Landmark indices per segment for YOLO (COCO 17 keypoints)
YOLO_LANDMARK_GROUPS = {
    'head': [0, 1, 2, 3, 4],       # Nose, eyes, ears
    'torso': [5, 6, 11, 12],       # Shoulders and hips
    'left_upper_arm': [5, 7],      # Shoulder to elbow
    'right_upper_arm': [6, 8],
    'left_lower_arm': [7, 9],      # Elbow to wrist (hand mass included)
    'right_lower_arm': [8, 10],
    'left_upper_leg': [11, 13],    # Hip to knee
    'right_upper_leg': [12, 14],
    'left_lower_leg': [13, 15],    # Knee to ankle
    'right_lower_leg': [14, 16]
}

def create_mass_vector(body_mass, yolo=False):

    dim = 17 if yolo else 33
    masses = np.zeros(dim)
    if body_mass is None:
        return masses

    # Assign mass proportionally to each landmark in a segment
    groups = YOLO_LANDMARK_GROUPS if yolo else LANDMARK_GROUPS
    for segment, indices in groups.items():
        fraction = MASS_FRACTIONS.get(segment, 0)
        segment_mass = body_mass * fraction
        per_landmark_mass = segment_mass / len(indices)
        for idx in indices:
            masses[idx] += per_landmark_mass
    return masses

def create_mass_dict(masses, group_names, use_table=False, yolo=False):

    masses_dict = {f"{name}_m": [] for name in group_names}

    if yolo:
        # Assign YOLO-specific groupings
        if use_table:
            masses_dict['whole_m'] = masses
            masses_dict['upper_m'] = masses[YoloBodyLandmarkGroups.UPPER_BODY]
            masses_dict['lower_m'] = masses[YoloBodyLandmarkGroups.LOWER_BODY]
            masses_dict['r_arm_m'] = masses[YoloBodyLandmarkGroups.RIGHT_ARM]
            masses_dict['l_arm_m'] = masses[YoloBodyLandmarkGroups.LEFT_ARM]
            masses_dict['r_leg_m'] = masses[YoloBodyLandmarkGroups.RIGHT_LEG]
            masses_dict['l_leg_m'] = masses[YoloBodyLandmarkGroups.LEFT_LEG]
        else:
            # Fallback: uniform mass
            masses_dict['whole_m'] = np.ones(len(YoloBodyLandmarkGroups.ALL))
            masses_dict['upper_m'] = np.ones(len(YoloBodyLandmarkGroups.UPPER_BODY))
            masses_dict['lower_m'] = np.ones(len(YoloBodyLandmarkGroups.LOWER_BODY))
            masses_dict['r_arm_m'] = np.ones(len(YoloBodyLandmarkGroups.RIGHT_ARM))
            masses_dict['l_arm_m'] = np.ones(len(YoloBodyLandmarkGroups.LEFT_ARM))
            masses_dict['r_leg_m'] = np.ones(len(YoloBodyLandmarkGroups.RIGHT_LEG))
            masses_dict['l_leg_m'] = np.ones(len(YoloBodyLandmarkGroups.LEFT_LEG))
    else:
        # MediaPipe groups
        if use_table:
            masses_dict['whole_m'] = masses
            masses_dict['upper_m'] = masses[BodyLandmarkGroups.UPPER_BODY]
            masses_dict['lower_m'] = masses[BodyLandmarkGroups.LOWER_BODY]
            masses_dict['r_arm_m'] = masses[BodyLandmarkGroups.RIGHT_ARM]
            masses_dict['l_arm_m'] = masses[BodyLandmarkGroups.LEFT_ARM]
            masses_dict['r_leg_m'] = masses[BodyLandmarkGroups.RIGHT_LEG]
            masses_dict['l_leg_m'] = masses[BodyLandmarkGroups.LEFT_LEG]
        else:
            # Fallback: uniform mass
            masses_dict['whole_m'] = np.ones(len(BodyLandmarkGroups.ALL))
            masses_dict['upper_m'] = np.ones(len(BodyLandmarkGroups.UPPER_BODY))
            masses_dict['lower_m'] = np.ones(len(BodyLandmarkGroups.LOWER_BODY))
            masses_dict['r_arm_m'] = np.ones(len(BodyLandmarkGroups.RIGHT_ARM))
            masses_dict['l_arm_m'] = np.ones(len(BodyLandmarkGroups.LEFT_ARM))
            masses_dict['r_leg_m'] = np.ones(len(BodyLandmarkGroups.RIGHT_LEG))
            masses_dict['l_leg_m'] = np.ones(len(BodyLandmarkGroups.LEFT_LEG))

    return masses_dict

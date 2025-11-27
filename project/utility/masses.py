import numpy as np
from classes.body_landmarks import BodyLandmarkGroups

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
    'right_lower_leg': 0.0465,
}

# Landmark groups per segment (MediaPipe Pose Landmarks indices)
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
    'right_lower_leg': [26, 28, 30, 32],
}

def create_mass_vector(body_mass):
    masses = np.zeros(33)
    if body_mass is None:
        return masses
    for segment, indices in LANDMARK_GROUPS.items():
        fraction = MASS_FRACTIONS.get(segment, 0)
        segment_mass = body_mass * fraction
        per_landmark_mass = segment_mass / len(indices)
        for idx in indices:
            masses[idx] += per_landmark_mass
    return masses

def create_mass_dict(masses, group_names, use_table):
    masses_dict = {
        f"{name}_m": []
        for name in group_names
    }

    if use_table is not None:
        masses_dict['whole_m'].append(masses)
        masses_dict['upper_m'].append(masses[BodyLandmarkGroups.UPPER_BODY])
        masses_dict['lower_m'].append(masses[BodyLandmarkGroups.LOWER_BODY])
        masses_dict['r_arm_m'].append(masses[BodyLandmarkGroups.RIGHT_ARM])
        masses_dict['l_arm_m'].append(masses[BodyLandmarkGroups.LEFT_ARM])
        masses_dict['r_leg_m'].append(masses[BodyLandmarkGroups.RIGHT_LEG])
        masses_dict['l_leg_m'].append(masses[BodyLandmarkGroups.LEFT_LEG])
    else:
        masses_dict['whole_m'].append(np.ones(len(BodyLandmarkGroups.ALL)))
        masses_dict['upper_m'].append(np.ones(len(BodyLandmarkGroups.UPPER_BODY)))
        masses_dict['lower_m'].append(np.ones(len(BodyLandmarkGroups.LOWER_BODY)))
        masses_dict['r_arm_m'].append(np.ones(len(BodyLandmarkGroups.RIGHT_ARM)))
        masses_dict['l_arm_m'].append(np.ones(len(BodyLandmarkGroups.LEFT_ARM)))
        masses_dict['r_leg_m'].append(np.ones(len(BodyLandmarkGroups.RIGHT_LEG)))
        masses_dict['l_leg_m'].append(np.ones(len(BodyLandmarkGroups.LEFT_LEG)))

    return masses_dict
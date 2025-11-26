import numpy as np

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
    for segment, indices in LANDMARK_GROUPS.items():
        fraction = MASS_FRACTIONS.get(segment, 0)
        segment_mass = body_mass * fraction
        per_landmark_mass = segment_mass / len(indices)
        for idx in indices:
            masses[idx] += per_landmark_mass
    return masses
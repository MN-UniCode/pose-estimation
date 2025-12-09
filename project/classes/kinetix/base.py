from abc import ABC, abstractmethod

class Kinetix(ABC):
    def __init__(self, fps, plot_window_seconds, total_mass, landmark_groups):
        # Initialize core parameters
        self.fps = fps
        self.total_mass = total_mass

        self.frame_width = None
        self.frame_height = None

        self.prev_p = None
        self.maxlen = int(fps * plot_window_seconds)

        # Body groups used for KE computation
        self.group_names = [
            "whole", "upper", "lower",
            "r_arm", "l_arm", "r_leg", "l_leg"
        ]

        # Corresponding landmark indices
        self.lm_list = [
            landmark_groups.ALL,
            landmark_groups.UPPER_BODY,
            landmark_groups.LOWER_BODY,
            landmark_groups.RIGHT_ARM,
            landmark_groups.LEFT_ARM,
            landmark_groups.RIGHT_LEG,
            landmark_groups.LEFT_LEG,
        ]

    def compare_ke(self, ke, dominance_ratio=2):
        # Compare KE between limbs to detect a dominant moving part
        limbs = {
            "right arm": ke["r_arm_ke"],
            "left arm": ke["l_arm_ke"],
            "right leg": ke["r_leg_ke"],
            "left leg": ke["l_leg_ke"],
        }

        dominant = max(limbs, key=limbs.get)
        dom_val = limbs[dominant]

        if dom_val < 0.1:
            return ""

        others = [v for k, v in limbs.items() if k != dominant]

        if all(v == 0 for v in others):
            return ""

        if all(dom_val > dominance_ratio * v for v in others):
            return f"The {dominant} is moving a lot."

        return ""
    
    @abstractmethod
    def __call__(self, detector, filters, cap, max_ke, use_anthropometric_tables=False):
        pass
    
    @abstractmethod
    def compute_components_ke(self, detection, dt, masses_dict, filters=None, track_id=None, max_speed=1):
        pass
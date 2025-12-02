import cv2
import numpy as np

# Mediapipe
import mediapipe as mp
from .body_landmarks import BodyLandmarkGroups
import utility.masses as masses

# Utilities and data structures
import time
import sys
from collections import deque
from .drawer import Drawer

class Kinetix:
    def __init__(self, fps, plot_window_seconds, total_mass):
        self.frame_width = None
        self.frame_height = None
        self.total_mass = total_mass
        self.fps = fps

        self.maxlen = int(fps) * plot_window_seconds

        self.group_names = ["whole", "upper", "lower", "r_arm", "l_arm", "r_leg", "l_leg"]

        self.lm_list = [BodyLandmarkGroups.ALL,
                   BodyLandmarkGroups.UPPER_BODY,
                   BodyLandmarkGroups.LOWER_BODY,
                   BodyLandmarkGroups.RIGHT_ARM,
                   BodyLandmarkGroups.LEFT_ARM,
                   BodyLandmarkGroups.RIGHT_LEG,
                   BodyLandmarkGroups.LEFT_LEG]

        # Create dictionary
        self.ke_histories = {
            f"{name}_ke": deque(maxlen=self.maxlen)
            for name in self.group_names
        }

    def __call__(self, detector, filters, cap, max_ke = 12.0, use_anthropometric_tables = False):
        # Checking for possible errors
        if not cap.isOpened():
            print("Error in opening the video stream.")
            sys.exit()
        
        drawer = Drawer()

        prev_detection = None
        prev_time = None

        group_plot = self.group_names

        previous_message = ""

        dt_ms = 1000 / self.fps
        timestamp_ms = 0

        # Compute the masses
        masses_vector = masses.create_mass_vector(self.total_mass)
        masses_dict = masses.create_mass_dict(masses_vector, self.group_names, use_anthropometric_tables)

        curr_time = time.time()

        if prev_time is None:
            dt_seconds = 1.0 / self.fps
        else:
            dt_seconds = curr_time - prev_time

        prev_time = curr_time

        keymap = {
            ord('l'): ['r_arm', 'r_leg', 'l_arm', 'l_leg'],
            ord('w'): ['whole', 'upper', 'lower'],
            ord('b'): self.group_names
        }

        while True:
            # Getting current frame
            success, current_frame = cap.read()
            if not success:
                break

            self.frame_height, self.frame_width, _ = current_frame.shape

            # Resizing it
            current_frame = cv2.resize(current_frame, (self.frame_width, self.frame_height))

            # Running detection
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=current_frame)
            detection_result = detector.detect_for_video(mp_image, int(timestamp_ms))

            # Getting current time and coordinates of the selected landmark
            # curr_time = time.time()

            # Computing kinetic energy
            ke = self.compute_components_kinetic_energy(detection_result, prev_detection,
                                        dt_seconds, masses_dict, filters)

            for name in group_plot:
                self.ke_histories[f'{name}_ke'].append(ke[f'{name}_ke'])

            message = self.compare_kinetic_energy(ke)
            if message != "":
                previous_message = message

            # Updating
            prev_detection = detection_result
            # prev_time = curr_time

            # Plotting
            annotated_image = drawer.draw_landmarks_on_image(current_frame, detection_result)
            ke_graph_image = drawer.draw_cv_barchart(ke, group_plot, height=annotated_image.shape[0], max_value=max_ke)

            combined = drawer.stack_images_horizontal([annotated_image, ke_graph_image])

            text_banner = drawer.create_text_banner(previous_message, width=ke_graph_image.shape[1] + annotated_image.shape[1])

            final = cv2.vconcat([combined, text_banner])

            timestamp_ms += dt_ms

            # Showing the result
            cv2.imshow("Landmarks overall kinetic energy", final)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key in keymap:
                group_plot = keymap[key]

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


    def compute_components_kinetic_energy(
            self, current_detection_result, previous_detection_result,
            dt, masses,
            position_filters=None, max_speed=1):

        ke = {f"{name}_ke": 0.0 for name in self.group_names}

        # Validate
        if (current_detection_result is None or previous_detection_result is None or
                not current_detection_result.pose_world_landmarks or
                not previous_detection_result.pose_world_landmarks):
            return ke

        curr_lm = current_detection_result.pose_world_landmarks[0]
        prev_lm = previous_detection_result.pose_world_landmarks[0]

        # Positions - 33x3
        curr_p = np.array([[lm.x, lm.y, lm.z] for lm in curr_lm])
        prev_p = np.array([[lm.x, lm.y, lm.z] for lm in prev_lm])

        # Filter POSITIONS, not velocities
        if position_filters is not None:
            for pos_filter in position_filters:
                curr_p = pos_filter.filter(curr_p.reshape(-1)).reshape(curr_p.shape)
                prev_p = pos_filter.filter(prev_p.reshape(-1)).reshape(prev_p.shape)

        # Velocities
        velocities = (curr_p - prev_p) / dt

        # Outlier removal
        speed = np.linalg.norm(velocities, axis=1)
        velocities[speed > max_speed] = 0

        # Remove body global translation (use midhip)
        midhip_curr = (curr_p[23] + curr_p[24]) / 2
        midhip_prev = (prev_p[23] + prev_p[24]) / 2
        root_vel = (midhip_curr - midhip_prev) / dt
        velocities = velocities - root_vel

        # KE for all groups
        for name, idx_group in zip(self.group_names, self.lm_list):
            v = velocities[idx_group]
            if len(v) == 0: continue

            group_mass = masses[f"{name}_m"]
            # point_mass = np.sum(group_mass) / len(v)

            ke[f"{name}_ke"] = 0.5 * np.sum(group_mass * np.sum(v ** 2, axis=1))

        return ke

    def compare_kinetic_energy(self, ke, dominance_ratio=2):
        relevant_groups = {
            "right arm": ke["r_arm_ke"],
            "left arm": ke["l_arm_ke"],
            "right leg": ke["r_leg_ke"],
            "left leg": ke["l_leg_ke"]
        }

        dominant_group = max(relevant_groups, key=relevant_groups.get)
        dominant_value = relevant_groups[dominant_group]
        if dominant_value < 0.1:
            return ""

        other_values = [v for k, v in relevant_groups.items() if k != dominant_group]

        if all(v == 0 for v in other_values):
            return ""

        if all(dominant_value > dominance_ratio * v for v in other_values):
            return f"The {dominant_group} is moving a lot."

        return ""

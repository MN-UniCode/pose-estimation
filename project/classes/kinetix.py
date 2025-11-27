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
    def __init__(self, fps, plot_window_seconds, frame_width, frame_height, total_mass):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.total_mass = total_mass

        self.maxlen = fps * plot_window_seconds

        self.group_names = ["whole", "upper", "lower", "r_arm", "l_arm", "r_leg", "l_leg"]

        # Create dictionary
        self.ke_histories = {
            f"{name}_ke": deque(maxlen=self.maxlen)
            for name in self.group_names
        }

    def __call__(self, detector, filter, cap, max_ke = 12.0, use_anthropometric_tables = False):
        # Checking for possible errors
        if not cap.isOpened():
            print("Error in opening the video stream.")
            sys.exit()
        
        drawer = Drawer()

        prev_detection = None
        prev_time = None

        group_plot = self.group_names

        previous_message = ""

        while True:
            # Getting current frame
            success, current_frame = cap.read()
            if not success:
                break

            # Resizing it
            current_frame = cv2.resize(current_frame, (self.frame_width, self.frame_height))

            # Running detection
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=current_frame)
            detection_result = detector.detect(mp_image)

            # Getting current time and coordinates of the selected landmark
            curr_time = time.time()

            # Computing kinetic energy
            masses_vector = masses.create_mass_vector(self.total_mass)
            masses_dict = masses.create_mass_dict(masses_vector, self.group_names, use_anthropometric_tables)

            ke = self.compute_components_kinetic_energy(detection_result, prev_detection,
                                        prev_time, curr_time, masses_dict, filter)

            for name in group_plot:
                self.ke_histories[f'{name}_ke'].append(ke[f'{name}_ke'])

            message = self.compare_kinetic_energy(ke)
            if message != "":
                previous_message = message

            # Updating
            prev_detection = detection_result
            prev_time = curr_time

            # Plotting
            annotated_image = drawer.draw_landmarks_on_image(current_frame, detection_result)
            ke_graph_image = drawer.draw_cv_barchart(ke, group_plot, annotated_image.shape[1], annotated_image.shape[0], max_ke)

            combined = drawer.stack_images_horizontal([annotated_image, ke_graph_image])

            text_banner = drawer.create_text_banner(previous_message, width=ke_graph_image.shape[1] + annotated_image.shape[1])

            final = cv2.vconcat([combined, text_banner])

            # Showing the result
            cv2.imshow("Landmarks overall kinetic energy", final)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                group_plot = ['r_arm', 'r_leg']
            elif key == ord('l'):
                group_plot = ['l_arm', 'l_leg']
            elif key == ord('w'):
                group_plot = ['whole', 'upper', 'lower']
            elif key == ord('b'):
                group_plot = self.group_names

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    # Computing the first-order derivative
    def first_order_derivative(self, curr_value, prev_value, curr_time, prev_time):
        result = None
        if curr_value is not None and prev_value is not None and curr_time is not None and prev_time is not None:
            dt = curr_time - prev_time
            result = (curr_value - prev_value) / dt
        return result

    def compute_components_kinetic_energy(self, current_detection_result, previous_detection_result,
                                          curr_time, prev_time,
                                          masses, velocity_filter=None):
        ke = {
            f"{name}_ke": 0.0
            for name in self.group_names
        }

        # Ensure landmarks exist
        if current_detection_result is None or previous_detection_result is None:
            return ke

        if previous_detection_result.pose_landmarks is None or len(previous_detection_result.pose_landmarks) == 0:
            return ke

        if current_detection_result.pose_landmarks is None or len(current_detection_result.pose_landmarks) == 0:
            return ke

        # Use only the first detected person
        current_landmarks = current_detection_result.pose_landmarks[0]
        previous_landmarks = previous_detection_result.pose_landmarks[0]

        n_landmarks = len(current_landmarks)

        # Compute velocity vectors for each landmark
        velocities = np.array([
            self.first_order_derivative(np.array([curr_lm.x, curr_lm.y, curr_lm.z]),
                                        np.array([prev_lm.x, prev_lm.y, prev_lm.z]), 
                                        curr_time, prev_time)
            for curr_lm, prev_lm in zip(current_landmarks, previous_landmarks)
        ])

        # Optional filtering
        if velocity_filter is not None:
            # Flatten velocities for filtering: (n_points * 3)
            v_flat = velocities.reshape(-1)
            # Filter one "sample" per channel (vectorized)
            v_f = velocity_filter.filter(v_flat)
            # Reshape the velocities as 3d array
            velocities = v_f.reshape(n_landmarks, 3)

        whole_v = velocities
        upper_v = velocities[BodyLandmarkGroups.UPPER_BODY]
        lower_v = velocities[BodyLandmarkGroups.LOWER_BODY]

        r_arm_v = velocities[BodyLandmarkGroups.RIGHT_ARM]
        l_arm_v = velocities[BodyLandmarkGroups.LEFT_ARM]

        r_leg_v = velocities[BodyLandmarkGroups.RIGHT_LEG]
        l_leg_v = velocities[BodyLandmarkGroups.LEFT_LEG]

        variables = [whole_v, upper_v, lower_v, r_arm_v, l_arm_v, r_leg_v, l_leg_v]

        for name, velocity in zip(self.group_names, variables):
            ke[f"{name}_ke"] = 0.5 * np.sum(masses[f"{name}_m"] * np.sum(velocity ** 2, axis=1))

        return ke

    def compare_kinetic_energy(self, ke, dominance_ratio=3):
        relevant_groups = {
            "right arm": ke["r_arm_ke"],
            "left arm": ke["l_arm_ke"],
            "right leg": ke["r_leg_ke"],
            "left leg": ke["l_leg_ke"]
        }

        dominant_group = max(relevant_groups, key=relevant_groups.get)
        dominant_value = relevant_groups[dominant_group]

        other_values = [v for k, v in relevant_groups.items() if k != dominant_group]

        if all(v == 0 for v in other_values):
            return ""

        if all(dominant_value > dominance_ratio * v for v in other_values):
            return f"The {dominant_group} is moving a lot."

        return ""

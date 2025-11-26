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
            if use_anthropometric_tables:
                masses_vector = masses.create_mass_vector(self.total_mass)
            else:
                masses_vector = None

            ke = self.compute_components_kinetic_energy(detection_result, prev_detection,
                                        prev_time, curr_time, masses_vector, filter)
            if ke is not None:
                for name in self.group_names:
                    self.ke_histories[f'{name}_ke'].append(ke[f'{name}_ke'])
            else:
                for name in self.group_names:
                    self.ke_histories[f'{name}_ke'].append(0.0)

            # Updating
            prev_detection = detection_result
            prev_time = curr_time

            # Plotting
            annotated_image = drawer.draw_landmarks_on_image(current_frame, detection_result)
            ke_graph_image = drawer.draw_cv_barchart(self.ke_histories, self.group_names, annotated_image.shape[1], annotated_image.shape[0], max_ke)
            combined = drawer.stack_images_horizontal([annotated_image, ke_graph_image])

            # Showing the result
            cv2.imshow("Landmarks overall kinetic energy", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
                                          masses=None, velocity_filter=None):
        # Ensure landmarks exist
        if current_detection_result is None or previous_detection_result is None:
            return None

        if previous_detection_result.pose_landmarks is None or len(previous_detection_result.pose_landmarks) == 0:
            return None

        if current_detection_result.pose_landmarks is None or len(current_detection_result.pose_landmarks) == 0:
            return None

        # Use only the first detected person
        current_landmarks = current_detection_result.pose_landmarks[0]
        previous_landmarks = previous_detection_result.pose_landmarks[0]

        n_landmarks = len(current_landmarks)
        if masses is None:
            masses = np.ones(n_landmarks)

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

        # TODO: group the masses to compute the kinetic energy of each component

        ke = {
            f"{name}_ke": 0.5 * np.sum(masses * np.sum(velocity ** 2, axis=1))
            for name, velocity in zip(self.group_names, variables)
        }

        return ke

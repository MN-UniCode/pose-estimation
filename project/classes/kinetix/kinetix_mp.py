import cv2
import numpy as np
import mediapipe as mp
import utility.masses as masses
import time
import sys
from ..drawer import Drawer
from .base import Kinetix


class Kinetix_mp(Kinetix):
    def __call__(self, detector, filters, cap, max_ke, sub_height=None, use_anthropometric_tables=False):
        # Ensure camera or video source is valid
        if not cap.isOpened():
            print("Error in opening the video stream.")
            sys.exit()

        # Prepare masses for each body group
        masses_vector = masses.create_mass_vector(self.total_mass)
        masses_dict = masses.create_mass_dict(
            masses_vector, self.group_names, use_anthropometric_tables
        )

        frame_index = 0
        prev_time = time.time()
        ms_unit = 1000.0
        previous_message = ""

        # Keyboard shortcuts for plot filtering
        keymap = {
            ord("l"): ["r_arm", "r_leg", "l_arm", "l_leg"],
            ord("w"): ["whole", "upper", "lower"],
            ord("b"): self.group_names,
        }

        plot = Drawer()
        group_plot = self.group_names

        while True:
            success, frame = cap.read()
            if not success:
                break

            self.frame_height, self.frame_width, _ = frame.shape

            # Sync frame timestamp with model
            timestamp_ms = int((ms_unit / self.fps) * frame_index)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame
            )
            detection = detector.detect_for_video(mp_image, timestamp_ms)

            # Compute time delta
            curr_time = time.time()
            dt = curr_time - prev_time
            dt = dt if 0 < dt <= 1.0 else 1.0 / self.fps
            prev_time = curr_time

            # Compute kinetic energy for each body part
            ke = self.compute_components_ke(detection, dt, masses_dict, filters)

            # Detect dominant movement
            message = self.compare_ke(ke)
            if message:
                previous_message = message

            plot(
                ke = ke, 
                max_ke = max_ke,
                message = previous_message, 
                group_plot = group_plot,
                frame = frame,
                detection = detection
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"): break
            if key in keymap: group_plot = keymap[key]

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

    def compute_components_ke(self, detection, dt, masses_dict, filters=None, max_speed=1):
        # Initialize KE dictionary
        ke = {f"{name}_ke": 0.0 for name in self.group_names}

        if not detection or not detection.pose_world_landmarks:
            return ke

        lm_list = detection.pose_world_landmarks[0]

        # Extract 3D landmark positions
        curr_p = np.array([[lm.x, lm.y, lm.z] for lm in lm_list])

        # Apply smoothing filters if provided
        if filters:
            for f in filters:
                curr_p = f.filter(curr_p.reshape(-1)).reshape(curr_p.shape)

        if self.prev_p is None:
            self.prev_p = curr_p

        # Compute landmark velocities
        velocities = (curr_p - self.prev_p) / dt

        # Ignore invisible landmarks
        visibility = np.array([lm.visibility for lm in lm_list]) > 0.5
        velocities[~visibility] = 0

        # Clamp unrealistic speeds
        speed = np.linalg.norm(velocities, axis=1)
        velocities[speed > max_speed] = 0

        # Compute KE for each body group
        for name, idx_group in zip(self.group_names, self.lm_list):
            v = velocities[idx_group]
            if len(v) == 0:
                continue

            mass = masses_dict[f"{name}_m"]
            ke[f"{name}_ke"] = 0.5 * np.sum(mass * np.sum(v * v, axis=1))

        self.prev_p = curr_p
        return ke

import cv2
import numpy as np
import time
import sys

from .base import Kinetix
from ..drawer import Drawer
import utility.masses as masses


class Kinetix_Yolo(Kinetix):
    # Main execution loop for YOLO-based tracking and KE computation
    def __call__(self, detector, filters, cap, max_ke, use_anthropometric_tables=False, sub_height=None,):
        if not cap.isOpened():
            print("Error in opening the video stream.")
            sys.exit()

        plot = Drawer()
        group_plot = self.group_names

        # Create mass vectors adapted for YOLO landmark count
        masses_vector = masses.create_mass_vector(self.total_mass, yolo=True)
        masses_dict = masses.create_mass_dict(masses_vector, self.group_names, use_anthropometric_tables, yolo=True)

        frame_index = 0
        prev_time = time.time()
        previous_message = ""

        # Group selection shortcuts
        keymap = {
            ord('l'): ['r_arm', 'r_leg', 'l_arm', 'l_leg'],
            ord('w'): ['whole', 'upper', 'lower'],
            ord('b'): self.group_names
        }

        while True:
            success, current_frame = cap.read()
            if not success:
                break

            self.frame_height, self.frame_width, _ = current_frame.shape

            # YOLOv8 tracking step
            results = detector.track(current_frame, verbose=False, persist=True, show=False)
            if len(results[0].keypoints) > 0:
                keypoints_data = results[0].keypoints.data[0].cpu().numpy()
            else:
                keypoints_data = None

            # Approximate world coordinates and scale
            world_kpts = self.yolo_to_world_approx(keypoints_data, subject_height_m=sub_height)

            # Time delta computation
            curr_time = time.time()
            dt = curr_time - prev_time
            dt = dt if 0 < dt <= 1.0 else 1.0 / self.fps
            prev_time = curr_time

            # Compute kinetic energy for each body group
            ke = self.compute_components_ke(world_kpts, dt, masses_dict, filters)

            # Compare KE to detect dominant movement
            message = self.compare_ke(ke)
            if message != "":
                previous_message = message

            # Retrieve YOLO-annotated frame
            annotated_image = results[0].plot()

            plot(
                ke=ke,
                max_ke=max_ke,
                message=previous_message,
                group_plot=group_plot,
                annotated_image=annotated_image
            )

            # Keyboard input handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in keymap:
                group_plot = keymap[key]

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

    # Compute kinetic energy for all tracked body components
    def compute_components_ke(self, detection, dt, masses_dict, filters=None, max_speed=1):
        ke = {f"{name}_ke": 0.0 for name in self.group_names}

        if detection is None:
            return ke

        # Extract 2D pixel coordinates
        curr_xy = detection[:, :2]
        curr_conf = detection[:, 2]

        # Build 3D array with z = 0 for compatibility
        curr_p = np.zeros((17, 3))
        curr_p[:, :2] = curr_xy

        # Apply optional filtering
        if filters is not None:
            for pos_filter in filters:
                curr_p = pos_filter.filter(curr_p.reshape(-1)).reshape(curr_p.shape)

        if self.prev_p is None:
            self.prev_p = curr_p

        # Compute velocities
        velocities = (curr_p - self.prev_p) / dt

        # Confidence-based visibility masking
        visible = curr_conf > 0.5
        velocities[~visible] = 0

        # Remove unrealistically fast movements
        speed = np.linalg.norm(velocities, axis=1)
        velocities[speed > max_speed] = 0

        # Compute KE for each group
        for name, idx_group in zip(self.group_names, self.lm_list):
            v = velocities[idx_group]
            if len(v) == 0:
                continue

            group_mass = masses_dict[f"{name}_m"]
            ke[f"{name}_ke"] = 0.5 * np.sum(group_mass * np.sum(v ** 2, axis=1))

        self.prev_p = curr_p

        return ke

    # Convert YOLO keypoints into approximate world coordinates using body height
    def yolo_to_world_approx(self, keypoints, subject_height_m=1.75):
        if keypoints is None:
            return self.last_world_kpts if hasattr(self, 'last_world_kpts') else np.zeros((17, 3))

        xy = keypoints[:, :2]
        conf = keypoints[:, 2]

        scale_factor = 0
        conf_thresh = 0.5

        # Estimate top of body (nose/eyes)
        top_y = None
        if conf[0] > conf_thresh:
            top_y = xy[0, 1]
        elif conf[1] > conf_thresh and conf[2] > conf_thresh:
            top_y = (xy[1, 1] + xy[2, 1]) / 2

        # Estimate lower body (ankles)
        ankles_y = []
        if conf[15] > conf_thresh:
            ankles_y.append(xy[15, 1])
        if conf[16] > conf_thresh:
            ankles_y.append(xy[16, 1])

        # Estimate shoulders width if height unavailable
        shoulders_x = []
        if conf[5] > conf_thresh:
            shoulders_x.append(xy[5, 0])
        if conf[6] > conf_thresh:
            shoulders_x.append(xy[6, 0])

        # Height-based scale estimation
        if top_y is not None and len(ankles_y) > 0:
            pixel_h = abs(np.mean(ankles_y) - top_y)
            pixel_tot = pixel_h / 0.88
            if pixel_tot > 1:
                scale_factor = subject_height_m / pixel_tot

        # Width-based fallback
        elif len(shoulders_x) > 0:
            width_pixels = abs(shoulders_x[0] - shoulders_x[1])
            ratio_shoulders = 0.24
            if width_pixels > 20:
                estimated_total_pixels = width_pixels / ratio_shoulders
                scale_factor = subject_height_m / estimated_total_pixels

        # Use previous scale if none computed
        if scale_factor == 0 and hasattr(self, 'last_scale_factor'):
            scale_factor = self.last_scale_factor

        if scale_factor == 0:
            return self.last_world_kpts if hasattr(self, 'last_world_kpts') else np.ones((17, 3))

        # Convert pixel coords to meters
        xy_m = xy * scale_factor

        world_kpts = np.zeros((17, 3))
        world_kpts[:, 0] = xy_m[:, 0]
        world_kpts[:, 1] = xy_m[:, 1]
        world_kpts[:, 2] = conf

        # Cache results for fallback
        self.last_scale_factor = scale_factor
        self.last_world_kpts = world_kpts

        return world_kpts
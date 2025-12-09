import cv2
import numpy as np
import time
import sys

from .base import Kinetix
from ..drawer import Drawer
import utility.masses as masses


class Kinetix_Yolo(Kinetix):
    def __init__(self, fps, plot_window_seconds, total_mass, landmark_groups, sub_height_m=1.75):
        super().__init__(fps, plot_window_seconds, total_mass, landmark_groups, sub_height_m)
        # Dizionari per gestire lo stato di ogni singola persona (Key: track_id)
        self.prev_p_dict = {}
        self.scale_factors_dict = {}
        self.last_world_kpts_dict = {}
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
            frame_ke_data = {}

            # Time delta computation
            curr_time = time.time()
            dt = curr_time - prev_time
            dt = dt if 0 < dt <= 1.0 else 1.0 / self.fps
            prev_time = curr_time

            if results[0].boxes.id is not None:
                # Ottieni ID e Keypoints
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints_all = results[0].keypoints.data.cpu().numpy()  # Shape (N, 17, 3)

                # Identifica quali ID sono presenti nel frame corrente
                current_ids_set = set(track_ids)

                # Iteriamo su ogni persona rilevata
                for i, track_id in enumerate(track_ids):
                    kpts_person = keypoints_all[i]  # Keypoints della singola persona

                    # 1. Converti in coordinate mondo (specifico per ID)
                    world_kpts = self.yolo_to_world_approx_id(kpts_person, track_id, subject_height_m=sub_height)

                    # 2. Calcola energia cinetica (specifico per ID)
                    ke_person = self.compute_components_ke(world_kpts, dt, masses_dict, filters, track_id)

                    frame_ke_data[track_id] = ke_person

                # Pulizia: Rimuovi dallo stato (prev_p) le persone che non sono più nel frame
                # Questo evita che se una persona esce e rientra dopo 10 secondi, venga calcolata una velocità assurda
                ids_to_remove = [k for k in self.prev_p_dict if k not in current_ids_set]
                for k in ids_to_remove:
                    del self.prev_p_dict[k]
                    # Nota: non cancelliamo scale_factor e last_world_kpts per mantenere la memoria della "taglia" se rientra subito
            else:
                # Nessuna persona rilevata, puliamo lo stato di velocità
                self.prev_p_dict = {}

            dominant_id = None
            max_total_ke = -1
            message = ""

            if frame_ke_data:
                # Trova chi ha la somma totale di KE più alta
                for t_id, ke_vals in frame_ke_data.items():
                    total_ke = sum(ke_vals.values())
                    if total_ke > max_total_ke:
                        max_total_ke = total_ke
                        dominant_id = t_id

                # Genera il messaggio basandosi solo sulla persona dominante
                if dominant_id is not None:
                    # Assumiamo che compare_ke sia metodo della classe base che restituisce stringa
                    message = self.compare_ke(frame_ke_data[dominant_id])
                    if message != "":
                        previous_message = message

            # Se non c'è nessuno, usa il messaggio precedente
            final_message = message if message != "" else previous_message

            # Retrieve YOLO-annotated frame
            annotated_image = results[0].plot()

            plot(
                ke=frame_ke_data,
                max_ke=max_ke,
                message=previous_message,
                group_plot=group_plot,
                annotated_image=annotated_image,
                dominant_id=dominant_id
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
    def compute_components_ke(self, detection, dt, masses_dict, filters=None, track_id=None, max_speed=1):
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

        prev_p = self.prev_p_dict.get(track_id, curr_p)

        # Compute velocities
        velocities = (curr_p - prev_p) / dt

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

        self.prev_p_dict[track_id] = curr_p

        return ke

    # Convert YOLO keypoints into approximate world coordinates using body height
    def yolo_to_world_approx_id(self, keypoints, track_id, subject_height_m=1.75):
        # if keypoints is None:
        #     return self.last_world_kpts if hasattr(self, 'last_world_kpts') else np.zeros((17, 3))

        xy = keypoints[:, :2]
        conf = keypoints[:, 2]

        # Recupera fattori di scala salvati per questo ID
        last_scale = self.scale_factors_dict.get(track_id, 0)
        last_world = self.last_world_kpts_dict.get(track_id, np.zeros((17, 3)))

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
        elif len(shoulders_x) == 2:
            width_pixels = abs(shoulders_x[0] - shoulders_x[1])
            ratio_shoulders = 0.24
            if width_pixels > 20:
                estimated_total_pixels = width_pixels / ratio_shoulders
                scale_factor = subject_height_m / estimated_total_pixels

        # Use previous scale if none computed
        if scale_factor == 0:
            scale_factor = last_scale

        if scale_factor == 0:
            return last_world

        # Convert pixel coords to meters
        xy_m = xy * scale_factor

        world_kpts = np.zeros((17, 3))
        world_kpts[:, 0] = xy_m[:, 0]
        world_kpts[:, 1] = xy_m[:, 1]
        world_kpts[:, 2] = conf

        # Salva stato
        self.scale_factors_dict[track_id] = scale_factor
        self.last_world_kpts_dict[track_id] = world_kpts

        return world_kpts
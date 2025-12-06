import cv2
import numpy as np
import time
import sys
from collections import deque

from .body_landmarks import YoloBodyLandmarkGroups
from .drawer import Drawer
import utility.masses as masses


class Kinetix:
    def __init__(self, fps, plot_window_seconds, total_mass):
        self.frame_width = None
        self.frame_height = None
        self.total_mass = total_mass
        self.fps = fps

        self.last_scale_factor = None
        self.last_world_kpts = None

        self.prev_p = None
        self.maxlen = int(fps) * plot_window_seconds

        self.group_names = ["whole", "upper", "lower", "r_arm", "l_arm", "r_leg", "l_leg"]

        # === Indici YOLO (COCO Format 17 Keypoints) ===

        self.lm_list = [YoloBodyLandmarkGroups.ALL,
                        YoloBodyLandmarkGroups.UPPER_BODY,
                        YoloBodyLandmarkGroups.LOWER_BODY,
                        YoloBodyLandmarkGroups.RIGHT_ARM,
                        YoloBodyLandmarkGroups.LEFT_ARM,
                        YoloBodyLandmarkGroups.RIGHT_LEG,
                        YoloBodyLandmarkGroups.LEFT_LEG]

        self.ke_histories = {
            f"{name}_ke": deque(maxlen=self.maxlen)
            for name in self.group_names
        }

    def __call__(self, model, filters, cap, max_ke, sub_height, use_anthropometric_tables=False):
        if not cap.isOpened():
            print("Error in opening the video stream.")
            sys.exit()

        drawer = Drawer()
        group_plot = self.group_names

        # NOTA: Assicurati che utility.masses sia aggiornato per vettori di lunghezza 3 (braccia/gambe)
        # invece di 5 o 6 come in MediaPipe, altrimenti darà errore di shape.
        masses_vector = masses.create_mass_vector(self.total_mass, yolo=True)
        masses_dict = masses.create_mass_dict(masses_vector, self.group_names, use_anthropometric_tables, yolo=True)

        frame_index = 0
        prev_time = time.time()
        previous_message = ""

        keymap = {
            ord('l'): ['r_arm', 'r_leg', 'l_arm', 'l_leg'],
            ord('w'): ['whole', 'upper', 'lower'],
            ord('b'): self.group_names
        }

        DISPLAY_HEIGHT = 600

        while True:
            success, current_frame = cap.read()
            if not success:
                break

            self.frame_height, self.frame_width, _ = current_frame.shape

            # === Inferenza YOLO ===
            # Eseguiamo YOLO sul frame corrente
            results = model.track(current_frame, verbose=False, persist=True, show=False)

            # Estraiamo i keypoints (shape: 1, 17, 3 -> x, y, conf)
            # Prendiamo il primo detection [0]
            if len(results[0].keypoints) > 0:
                # Portiamo su CPU e convertiamo in numpy
                keypoints_data = results[0].keypoints.data[0].cpu().numpy()
            else:
                keypoints_data = None

            world_kpts = self.yolo_to_world_approx(keypoints_data, subject_height_m=sub_height)

            # Gestione del tempo
            curr_time = time.time()
            dt_seconds = curr_time - prev_time
            if dt_seconds <= 0 or dt_seconds > 1.0:
                dt_seconds = 1.0 / self.fps
            prev_time = curr_time

            # === Calcolo KE ===
            ke = self.compute_components_kinetic_energy(world_kpts,
                                                        dt_seconds, masses_dict, filters)

            for name in group_plot:
                self.ke_histories[f'{name}_ke'].append(ke[f'{name}_ke'])

            message = self.compare_kinetic_energy(ke)
            if message != "":
                previous_message = message

            # Plotting (YOLO ha un suo plotter integrato, ma usiamo il tuo drawer)
            # Nota: drawer.draw_landmarks_on_image è fatto per oggetti MediaPipe.
            # Per YOLO dovrai usare results[0].plot() o adattare il drawer.
            # Qui usiamo il plot nativo di YOLO per semplicità sul frame:
            annotated_image = results[0].plot()

            # Resize e visualizzazione (Logica invariata)
            aspect_ratio_video = annotated_image.shape[1] / annotated_image.shape[0]
            display_w = int(DISPLAY_HEIGHT * aspect_ratio_video)
            annotated_display = cv2.resize(annotated_image, (display_w, DISPLAY_HEIGHT))

            ke_graph_image = drawer.draw_cv_barchart(ke, group_plot, target_height=DISPLAY_HEIGHT, max_value=max_ke)
            combined = drawer.stack_images_horizontal([annotated_display, ke_graph_image])

            total_width = combined.shape[1]
            text_banner = drawer.create_text_banner(previous_message, width=total_width, height=80)

            if text_banner.shape[1] != combined.shape[1]:
                text_banner = cv2.resize(text_banner, (combined.shape[1], 80))

            final = cv2.vconcat([combined, text_banner])

            cv2.imshow("Landmarks overall kinetic energy", final)

            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key in keymap: group_plot = keymap[key]

        cap.release()
        cv2.destroyAllWindows()

    def compute_components_kinetic_energy(
            self, keypoints,  # Ora riceve un array numpy (17, 3)
            dt, masses,
            position_filters=None, max_speed=5000):  # Aumentato max_speed perché siamo in pixel

        ke = {f"{name}_ke": 0.0 for name in self.group_names}

        # Validazione input YOLO
        if keypoints is None:
            return ke

        # === MODIFICA 4: Coordinate e Assi ===
        # YOLO restituisce [x, y, conf]. MediaPipe dava x,y,z normalizzati o in metri.
        # Qui usiamo i pixel. Z non c'è, quindi lo mettiamo a 0.

        # Estraiamo x, y
        curr_xy = keypoints[:, :2]
        # Estraiamo confidenza
        curr_conf = keypoints[:, 2]

        # Creiamo un vettore (17, 3) dove z=0 per compatibilità
        curr_p = np.zeros((17, 3))
        curr_p[:, :2] = curr_xy

        # Filtro POSIZIONI
        if position_filters is not None:
            for pos_filter in position_filters:
                curr_p = pos_filter.filter(curr_p.reshape(-1)).reshape(curr_p.shape)

        if self.prev_p is None:
            self.prev_p = curr_p

        # Calcolo Velocità
        velocities = (curr_p - self.prev_p) / dt

        # Filtro visibilità (usiamo la confidenza di YOLO)
        visible = curr_conf > 0.5
        velocities[~visible] = 0

        # Outlier removal (Magnitude velocità)
        speed = np.linalg.norm(velocities, axis=1)
        velocities[speed > max_speed] = 0

        # KE for all groups
        for name, idx_group in zip(self.group_names, self.lm_list):
            v = velocities[idx_group]
            if len(v) == 0: continue

            group_mass = masses[f"{name}_m"]
            # point_mass = np.sum(group_mass) / len(v)

            ke[f"{name}_ke"] = 0.5 * np.sum(group_mass * np.sum(v ** 2, axis=1))

        self.prev_p = curr_p

        return ke

    def yolo_to_world_approx(self, keypoints, subject_height_m=1.75):

        if keypoints is None:
            return self.last_world_kpts if self.last_world_kpts is not None else np.zeros((17, 3))

        xy = keypoints[:, :2]
        conf = keypoints[:, 2]

        scale_factor = 0
        conf_thresh = 0.5

        # Occhi/Naso (Top)
        top_y = None
        if conf[0] > conf_thresh:
            top_y = xy[0, 1]  # Naso
        elif conf[1] > conf_thresh and conf[2] > conf_thresh:
            top_y = (xy[1, 1] + xy[2, 1]) / 2  # Media Occhi

        # Caviglie (Bottom Full)
        ankles_y = []
        if conf[15] > conf_thresh: ankles_y.append(xy[15, 1])
        if conf[16] > conf_thresh: ankles_y.append(xy[16, 1])

        # Spalle (Per larghezza o altezza)
        shoulders_x = []
        if conf[5] > conf_thresh: shoulders_x.append(xy[5, 0])
        if conf[6] > conf_thresh: shoulders_x.append(xy[6, 0])

        if top_y is not None and len(ankles_y) > 0:
            pixel_h = abs(np.mean(ankles_y) - top_y)
            pixel_tot = pixel_h / 0.88
            if pixel_tot > 1:
                scale_factor = subject_height_m / pixel_tot
        elif len(shoulders_x) > 0:
            width_pixels = abs(shoulders_x[0] - shoulders_x[1])

            ratio_shoulders = 0.24

            if width_pixels > 20:
                estimated_total_pixels = width_pixels / ratio_shoulders
                scale_factor = subject_height_m / estimated_total_pixels


        # ---> fallback: usa comunque l'ultima scala valida
        if scale_factor is None:
            scale_factor = self.last_scale_factor

        # ---> se ancora nulla restituisce uni per i pixel
        if scale_factor is None:
            return self.last_world_kpts if self.last_world_kpts is not None else np.ones((17, 3))

        # Conversione in metri
        xy_m = xy * scale_factor

        world_kpts = np.zeros((17, 3))
        world_kpts[:, 0] = xy_m[:, 0]
        world_kpts[:, 1] = xy_m[:, 1]
        world_kpts[:, 2] = conf

        # salva per fallback
        self.last_scale_factor = scale_factor
        self.last_world_kpts = world_kpts

        return world_kpts

    def compare_kinetic_energy(self, ke, dominance_ratio=2):
        relevant_groups = {
            "right arm": ke["r_arm_ke"],
            "left arm": ke["l_arm_ke"],
            "right leg": ke["r_leg_ke"],
            "left leg": ke["l_leg_ke"]
        }

        dominant_group = max(relevant_groups, key=relevant_groups.get)
        dominant_value = relevant_groups[dominant_group]
        if dominant_value < 0.1:  # Soglia da ricalibrare poiché siamo in pixel^2/s^2
            return ""

        other_values = [v for k, v in relevant_groups.items() if k != dominant_group]

        if all(v == 0 for v in other_values):
            return ""

        if all(dominant_value > dominance_ratio * v for v in other_values):
            return f"The {dominant_group} is moving a lot."

        return ""
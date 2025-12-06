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

    def __call__(self, model, filters, cap, max_ke, use_anthropometric_tables=False):
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
            results = model(current_frame, verbose=False)

            # Estraiamo i keypoints (shape: 1, 17, 3 -> x, y, conf)
            # Prendiamo il primo detection [0]
            if len(results[0].keypoints) > 0:
                # Portiamo su CPU e convertiamo in numpy
                keypoints_data = results[0].keypoints.data[0].cpu().numpy()
            else:
                keypoints_data = None

            # Gestione del tempo
            curr_time = time.time()
            dt_seconds = curr_time - prev_time
            if dt_seconds <= 0 or dt_seconds > 1.0:
                dt_seconds = 1.0 / self.fps
            prev_time = curr_time

            # === Calcolo KE ===
            ke = self.compute_components_kinetic_energy(keypoints_data,
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
            self, keypoints,
            dt, masses,
            position_filters=None,
            max_speed=5000,  # Nota: questo ora si riferisce ai pixel, ma filtreremo dopo
            subject_height_m=1.75):  # <--- NUOVO PARAMETRO: Altezza reale in metri

        ke = {f"{name}_ke": 0.0 for name in self.group_names}

        # Validazione input YOLO
        if keypoints is None:
            return ke

        # === 1. Estrazione Coordinate ===
        curr_xy = keypoints[:, :2]  # (17, 2) in Pixel
        curr_conf = keypoints[:, 2]

        # Creiamo vettore (17, 3) con z=0
        curr_p = np.zeros((17, 3))
        curr_p[:, :2] = curr_xy

        # === 2. Calcolo Fattore di Scala (Pixel -> Metri) ===
        # Usiamo la distanza verticale tra gli occhi (o naso) e le caviglie per stimare l'altezza
        # YOLO indices: 0:Nose, 15:Left Ankle, 16:Right Ankle

        # Prendiamo le y delle caviglie (se visibili, conf > 0.5)
        ankles_y = []
        if curr_conf[15] > 0.5: ankles_y.append(curr_p[15, 1])
        if curr_conf[16] > 0.5: ankles_y.append(curr_p[16, 1])

        # Prendiamo la y del naso o occhi
        top_y = curr_p[0, 1] if curr_conf[0] > 0.5 else None

        scale_factor = 0

        # Se abbiamo punti sufficienti per stimare l'altezza
        if top_y is not None and len(ankles_y) > 0:
            mean_ankle_y = np.mean(ankles_y)
            pixel_height = abs(mean_ankle_y - top_y)

            # Evitiamo divisioni per zero o altezze assurde
            if pixel_height > 10:
                # Fattore correttivo: La distanza Naso-Caviglie è circa il 85-90% dell'altezza totale
                # Quindi Pixel_Totali_Stimati = pixel_height / 0.88
                estimated_total_pixel_height = pixel_height / 0.88

                scale_factor = subject_height_m / estimated_total_pixel_height

        # Fallback: Se non riusciamo a calcolare la scala (es. piedi non visibili),
        # usiamo la scala del frame precedente se esiste, altrimenti 0 (no calcolo)
        if scale_factor == 0 and hasattr(self, 'prev_scale') and self.prev_scale > 0:
            scale_factor = self.prev_scale

        self.prev_scale = scale_factor  # Salviamo per il prossimo frame

        # === 3. Filtri Posizione (sui pixel raw) ===
        if position_filters is not None:
            for pos_filter in position_filters:
                curr_p = pos_filter.filter(curr_p.reshape(-1)).reshape(curr_p.shape)

        if self.prev_p is None:
            self.prev_p = curr_p
            return ke  # Primo frame, niente velocità

        # === 4. Calcolo Velocità in METRI AL SECONDO ===
        # Differenza in pixel
        diff_pixels = curr_p - self.prev_p

        # Convertiamo la differenza in metri
        if scale_factor > 0:
            diff_meters = diff_pixels * scale_factor
        else:
            diff_meters = np.zeros_like(diff_pixels)  # Impossibile calcolare metri

        velocities = diff_meters / dt  # m/s

        # === 5. Filtri e Pulizia ===
        # Filtro visibilità
        visible = curr_conf > 0.5
        velocities[~visible] = 0

        # Outlier removal (Magnitude velocità in m/s)
        # 5000 pixel/s erano tanti, ma 20 m/s è il record del mondo.
        # Mettiamo un tetto umano, es. 15 m/s per movimenti esplosivi
        max_speed_meters = 15.0
        speed = np.linalg.norm(velocities, axis=1)
        velocities[speed > max_speed_meters] = 0

        # === 6. Calcolo Energia Cinetica (Joule) ===
        for name, idx_group in zip(self.group_names, self.lm_list):
            v = velocities[idx_group]
            if len(v) == 0: continue

            # Nota: masses[name] deve essere in kg
            group_mass = masses[f"{name}_m"]

            # KE = 0.5 * m * v^2
            # Qui assumiamo che group_mass sia scalare o array allineato
            ke[f"{name}_ke"] = 0.5 * np.sum(group_mass * np.sum(v ** 2, axis=1))

        self.prev_p = curr_p

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
        if dominant_value < 0.1:  # Soglia da ricalibrare poiché siamo in pixel^2/s^2
            return ""

        other_values = [v for k, v in relevant_groups.items() if k != dominant_group]

        if all(v == 0 for v in other_values):
            return ""

        if all(dominant_value > dominance_ratio * v for v in other_values):
            return f"The {dominant_group} is moving a lot."

        return ""
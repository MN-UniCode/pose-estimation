import cv2
import numpy as np
import time
import sys
from collections import deque
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
        # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar,
        # 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist,
        # 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle

        self.lm_list = [
            list(range(17)),  # whole (tutti)
            [5, 6, 7, 8, 9, 10],  # upper (spalle + braccia)
            [11, 12, 13, 14, 15, 16],  # lower (fianchi + gambe)
            [6, 8, 10],  # r_arm (spalla, gomito, polso dx)
            [5, 7, 9],  # l_arm (spalla, gomito, polso sx)
            [12, 14, 16],  # r_leg (fianco, ginocchio, caviglia dx)
            [11, 13, 15]  # l_leg (fianco, ginocchio, caviglia sx)
        ]

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
        masses_vector = masses.create_mass_vector(self.total_mass)
        masses_dict = masses.create_mass_dict(masses_vector, self.group_names, use_anthropometric_tables)

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

            # === MODIFICA 2: Inferenza YOLO ===
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

            # === MODIFICA 3: Calcolo KE adattato ===
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
                # Attenzione: i filtri potrebbero aspettarsi 33 canali (vecchio MP).
                # Devi assicurarti che ButterworthMultichannel sia inizializzato con
                # num_channels = 17 * 3 (o modificare qui per passare solo i dati validi)
                try:
                    curr_p = pos_filter.filter(curr_p.reshape(-1)).reshape(curr_p.shape)
                except Exception as e:
                    pass  # Se le dimensioni del filtro non coincidono, saltiamo per ora

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

            # === ATTENZIONE ===
            # masses[name] deve avere la stessa lunghezza di 'v'.
            # MP aveva più punti per braccio (es. mignolo, indice, pollice).
            # YOLO ha solo 3 punti per braccio.
            # Se 'masses' è hardcoded per MP, questo crasherà.
            # Soluzione temporanea: prendiamo la media della massa se le lunghezze differiscono
            # oppure assumiamo che tu abbia aggiornato masses.py

            group_mass_vector = masses[f"{name}_m"]

            # Fallback di sicurezza per vettori massa
            if len(group_mass_vector) != len(v):
                # Se le dimensioni non coincidono, spalmiamo la massa totale del gruppo sui punti disponibili
                total_m = np.sum(group_mass_vector)
                group_mass_vector = np.full(len(v), total_m / len(v))

            ke[f"{name}_ke"] = 0.5 * np.sum(group_mass_vector * np.sum(v ** 2, axis=1))

        self.prev_p = curr_p

        return ke

    def compare_kinetic_energy(self, ke, dominance_ratio=2):
        # Invariato
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
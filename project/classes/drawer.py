import cv2
import numpy as np

# Mediapipe
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2



class Drawer:
    def __init__(self):
        pass

    # Drawing body landmarks
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = rgb_image.copy()

        # Loop through the detected poses to visualize
        if len(pose_landmarks_list) > 0:
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]

                # Draw the pose landmarks
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList() # this has just x,y,z and not visibility and presence
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())

        return annotated_image

    def draw_yolo_landmarks_on_image(self, image, landmarks_33):
        """
        landmarks_33 : np.array di shape (33,3)
                       coordinate assolute in pixel (x,y,z ignorato)
        """

        annotated = image.copy()

        if landmarks_33 is None:
            return annotated

        # --- Punto
        for (x, y, z) in landmarks_33:
            cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)

        # --- Connessioni del modello MediaPipe POSE
        # La lista ufficiale (ridotta alle connessioni principali)
        mp_connections = [
            (11, 13), (13, 15),  # braccio sinistro
            (12, 14), (14, 16),  # braccio destro
            (23, 25), (25, 27),  # gamba sinistra
            (24, 26), (26, 28),  # gamba destra
            (11, 12),  # spalle
            (23, 24),  # anche
            (11, 23), (12, 24),  # torso
            (0, 11), (0, 12)  # testa→spalle
        ]

        for a, b in mp_connections:
            xa, ya, za = landmarks_33[a]
            xb, yb, zb = landmarks_33[b]
            cv2.line(annotated, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

        return annotated

    def draw_cv_barchart(self, ke, group_names, target_height=600, max_value=12.0):
        # Impostiamo l'altezza fissa desiderata per la visualizzazione
        height = target_height
        # La larghezza del grafico è proporzionata all'altezza (es. 16:9 o 4:3)
        # Invece di dipendere dal video, la facciamo dipendere dall'altezza del grafico stesso
        width = int(height * 1.5)

        graph = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas

        # Calcolo dinamico delle dimensioni in base all'altezza (scalatura)
        base_h = 600.0
        scale = height / base_h

        margin_left = int(100 * scale)
        margin_bottom = int(60 * scale)
        margin_top = int(20 * scale)

        font_scale = 0.8 * scale
        thickness = max(1, int(2 * scale))

        bar_width = int((width - margin_left - int(20 * scale)) / len(group_names))

        # Draw axes
        cv2.line(graph, (margin_left, margin_top), (margin_left, height - margin_bottom), (0, 0, 0), thickness)
        cv2.line(graph, (margin_left, height - margin_bottom), (width - 10, height - margin_bottom), (0, 0, 0),
                 thickness)

        values = []
        for name in group_names:
            hist = ke[f"{name}_ke"]
            values.append(hist)

        # Plot bars
        for i, (name, value) in enumerate(zip(group_names, values)):
            # Clamp value per evitare che la barra esca dal grafico
            draw_val = min(value, max_value)

            bar_height = int((draw_val / max_value) * (height - margin_bottom - margin_top))
            x1 = margin_left + i * bar_width + int(5 * scale)
            y1 = height - margin_bottom - bar_height
            x2 = x1 + bar_width - int(10 * scale)
            y2 = height - margin_bottom

            cv2.rectangle(graph, (x1, y1), (x2, y2), (0, 122, 255), -1)

            # Label group name
            text_y_offset = int(40 * scale)
            cv2.putText(graph, name.replace("_", ""),
                        (x1, height - margin_bottom + text_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            # Label value
            cv2.putText(graph, f"{value:.1f}",
                        (x1, y1 - int(5 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Y-axis label
        cv2.putText(graph, "K.E.", (int(5 * scale), margin_top + int(20 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), thickness)

        return graph

    # Modifichiamo anche stack_images per gestire il resize automatico se le altezze non combaciano
    def stack_images_horizontal(self, images):
        if not images:
            return None

        # Troviamo l'altezza del primo'immagine (che useremo come riferimento)
        h_target = images[0].shape[0]

        resized_images = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Se l'altezza è diversa, ridimensioniamo mantenendo l'aspect ratio
            if img.shape[0] != h_target:
                scale = h_target / img.shape[0]
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, h_target))

            resized_images.append(img)

        return cv2.hconcat(resized_images)

    def create_text_banner(self, text, width=640, height=140, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
        banner = np.ones((height, width, 3), dtype=np.uint8)
        banner[:] = bg_color

        # Center text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2
        thickness = 3
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]

        x = (width - text_size[0]) // 2
        y = (height + text_size[1]) // 2

        cv2.putText(banner, text, (x, y), font, scale, text_color, thickness)
        return banner

    # Stack OpenCV images horizontally
    # def stack_images_horizontal(self, images, scale=1.0):
    #     resized_images = []
    #     for img in images:
    #         if len(img.shape) == 2:  # grayscale
    #             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #         img = cv2.resize(img, None, fx=scale, fy=scale)
    #         resized_images.append(img)
    #     return cv2.hconcat(resized_images)
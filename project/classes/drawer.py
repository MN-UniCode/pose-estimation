import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class Drawer:
    def __init__(self):
        # Palette colori per diversi ID (BGR)
        self.colors = [
            (0, 122, 255),  # Arancione
            (0, 255, 0),    # Verde
            (255, 0, 0),    # Blu
            (0, 255, 255),  # Giallo
            (255, 0, 255),  # Magenta
            (255, 255, 0)   # Ciano
        ]

    def __call__(self, ke, max_ke, message, group_plot, frame = None, detection = None, annotated_image = None, dominant_id = None):
        # Resize main annotated image for consistent visualization
        display_h = 600
        mp = False

        if annotated_image is None:
            mp = True
            annotated_image = self._draw_landmarks_on_image(frame, detection)

        ratio = annotated_image.shape[1] / annotated_image.shape[0]
        disp_w = int(display_h * ratio)
        annotated_resized = cv2.resize(annotated_image, (disp_w, display_h))

        # Draw KE bar chart
        if mp:
            ke_chart = self._draw_cv_barchart(
                ke, group_plot, target_height=display_h, max_value=max_ke
            )
        else:
            ke_chart = self._draw_multi_barchart(
                ke, group_plot, target_height=display_h, max_value=max_ke
            )

        # Combine side-by-side
        combined = self._stack_images_horizontal([annotated_resized, ke_chart])

        text_color = (0, 0, 0)
        if dominant_id is not None:
            text_color = self.colors[dominant_id % len(self.colors)]

        # Display message banner
        banner = self._create_text_banner(message, width=combined.shape[1], height=80, text_color=text_color)
        
        if banner.shape[1] != combined.shape[1]:
            banner = cv2.resize(banner, (combined.shape[1], 80))

        final = cv2.vconcat([combined, banner])
        cv2.imshow("Landmarks overall kinetic energy", final)

    def _draw_landmarks_on_image(self, rgb_image, detection_result):
        # Draw Mediapipe pose landmarks when available
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = rgb_image.copy()

        if len(pose_landmarks_list) > 0:
            for pose_landmarks in pose_landmarks_list:
                # Convert to protobuf format for MediaPipe drawing utility
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in pose_landmarks
                ])

                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style()
                )

        return annotated_image

    def _draw_cv_barchart(self, ke, group_names, target_height=600, max_value=12.0):
        # Create bar chart canvas resized based on desired height
        height = target_height
        width = int(height * 1.5)
        graph = np.ones((height, width, 3), dtype=np.uint8) * 255

        scale = height / 600.0

        margin_left = int(100 * scale)
        margin_bottom = int(60 * scale)
        margin_top = int(20 * scale)

        font_scale = 0.8 * scale
        thickness = max(1, int(2 * scale))

        bar_width = int((width - margin_left - int(20 * scale)) / len(group_names))

        # Draw axes
        cv2.line(graph, (margin_left, margin_top), (margin_left, height - margin_bottom), (0, 0, 0), thickness)
        cv2.line(graph, (margin_left, height - margin_bottom), (width - 10, height - margin_bottom), (0, 0, 0), thickness)

        values = [ke[f"{name}_ke"] for name in group_names]

        # Draw each bar
        for i, (name, value) in enumerate(zip(group_names, values)):
            draw_val = min(value, max_value)  # prevent bars from exceeding chart

            bar_height = int((draw_val / max_value) * (height - margin_bottom - margin_top))

            x1 = margin_left + i * bar_width + int(5 * scale)
            y1 = height - margin_bottom - bar_height
            x2 = x1 + bar_width - int(10 * scale)
            y2 = height - margin_bottom

            cv2.rectangle(graph, (x1, y1), (x2, y2), (0, 122, 255), -1)

            # Write group name under bar
            cv2.putText(
                graph, name.replace("_", ""),
                (x1, height - margin_bottom + int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness
            )

            # Write KE value above bar
            cv2.putText(
                graph, f"{value:.1f}",
                (x1, y1 - int(5 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness
            )

        # Label Y-axis
        cv2.putText(graph, "K.E.", (int(5 * scale), margin_top + int(20 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), thickness)

        return graph

    def _draw_multi_barchart(self, all_ke_data, group_names, target_height=600, max_value=12.0):
        # Configurazione dimensioni
        height = target_height
        width = int(height * 1.5)
        graph = np.ones((height, width, 3), dtype=np.uint8) * 255  # Sfondo bianco

        scale = height / 600.0
        margin_left = int(80 * scale)
        margin_bottom = int(60 * scale)
        margin_top = int(40 * scale)

        # Spessore linee assi
        font_scale = 0.8 * scale
        thickness = max(1, int(2 * scale))

        # Disegna Assi
        cv2.line(graph, (margin_left, margin_top), (margin_left, height - margin_bottom), (0, 0, 0), thickness)
        cv2.line(graph, (margin_left, height - margin_bottom), (width - 10, height - margin_bottom), (0, 0, 0),
                 thickness)

        if not all_ke_data:
            cv2.putText(graph, "No Detection", (width // 2 - 50, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            return graph

        # Parametri barre
        n_groups = len(group_names)
        n_people = len(all_ke_data)
        track_ids = sorted(list(all_ke_data.keys()))  # Ordina ID per consistenza colore

        # Calcolo spazi
        group_width = (width - margin_left - 20) / n_groups
        group_padding = group_width * 0.2
        bar_width = (group_width - group_padding) / n_people

        for i, group_name in enumerate(group_names):
            # Posizione base del gruppo sull'asse X
            group_x_start = margin_left + (i * group_width) + (group_padding / 2)

            # Etichetta asse X (nome parte del corpo) centrata nel gruppo
            label_x = int(group_x_start + (group_width - group_padding) / 2 - 20 * scale)
            cv2.putText(graph, group_name.replace("_", ""),
                        (label_x, height - margin_bottom + int(30 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            # Disegna le barre per ogni persona in questo gruppo
            for j, track_id in enumerate(track_ids):
                # Ottieni valore KE
                ke_val = all_ke_data[track_id].get(f"{group_name}_ke", 0.0)
                draw_val = min(ke_val, max_value)

                # Calcola altezza barra
                bar_height = int((draw_val / max_value) * (height - margin_bottom - margin_top))

                # Coordinate rettangolo
                x1 = int(group_x_start + j * bar_width)
                y1 = height - margin_bottom - bar_height
                x2 = int(x1 + bar_width - 2)
                y2 = height - margin_bottom

                # Colore basato sull'ID
                color = self.colors[track_id % len(self.colors)]

                cv2.rectangle(graph, (x1, y1), (x2, y2), color, -1)

                # --- MODIFICA: Visualizzazione Valore ---
                if bar_width > 10:  # Scrivi solo se la barra è abbastanza larga
                    value_text = f"{ke_val:.1f}"

                    # Scala del font dinamica: se ci sono molte persone, riduci il font
                    font_val_scale = 0.4 * scale if n_people > 2 else 0.6 * scale

                    # Calcola dimensione testo per centrarlo
                    (text_w, text_h), _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, font_val_scale, 1)

                    # Centro della barra
                    bar_center_x = x1 + (x2 - x1) // 2
                    text_x = int(bar_center_x - text_w // 2)
                    text_y = y1 - 5  # 5 pixel sopra la barra

                    # Se la barra è troppo bassa (valore vicino a 0), alza il testo per non sovrapporlo alla linea nera
                    if text_y > height - margin_bottom - 5:
                        text_y = height - margin_bottom - 5

                    cv2.putText(graph, value_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_val_scale, (0, 0, 0), 1)

        # Etichetta asse Y
        cv2.putText(graph, "K.E. (J)", (5, margin_top), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 0), thickness)

        # Legenda ID in alto a destra
        for k, tid in enumerate(track_ids):
            c = self.colors[tid % len(self.colors)]
            cv2.putText(graph, f"ID {tid}", (width - int(100 * scale), margin_top + k * int(20 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, c, thickness)

        return graph

    def _stack_images_horizontal(self, images):
        # Resize images to same height while preserving aspect ratio
        if not images:
            return None

        h_target = images[0].shape[0]
        resized_images = []

        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if img.shape[0] != h_target:
                scale = h_target / img.shape[0]
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, h_target))

            resized_images.append(img)

        return cv2.hconcat(resized_images)

    def _create_text_banner(self, text, width=640, height=140, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
        # Create centered text banner used for messages
        banner = np.ones((height, width, 3), dtype=np.uint8)
        banner[:] = bg_color

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2
        thickness = 3
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]

        x = (width - text_size[0]) // 2
        y = (height + text_size[1]) // 2

        cv2.putText(banner, text, (x, y), font, scale, text_color, thickness)
        return banner
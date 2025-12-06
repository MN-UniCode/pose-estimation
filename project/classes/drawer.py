import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class Drawer:
    def __init__(self):
        pass

    def __call__(self, ke, max_ke, message, group_plot, frame = None, detection = None, annotated_image = None ):
        # Resize main annotated image for consistent visualization
        display_h = 600

        if annotated_image is None:
            annotated_image = self._draw_landmarks_on_image(frame, detection)

        ratio = annotated_image.shape[1] / annotated_image.shape[0]
        disp_w = int(display_h * ratio)
        annotated_resized = cv2.resize(annotated_image, (disp_w, display_h))

        # Draw KE bar chart
        ke_chart = self._draw_cv_barchart(
            ke, group_plot, target_height=display_h, max_value=max_ke
        )

        # Combine side-by-side
        combined = self._stack_images_horizontal([annotated_resized, ke_chart])

        # Display message banner
        banner = self._create_text_banner(message, width=combined.shape[1], height=80)
        
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

    def _draw_yolo_landmarks_on_image(self, image, landmarks_33):
        # Draw YOLO-generated pose landmarks + basic MediaPipe-style connections
        annotated = image.copy()

        if landmarks_33 is None:
            return annotated

        for (x, y, z) in landmarks_33:
            cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Connections approximating MediaPipe POSE edges
        mp_connections = [
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (23, 25), (25, 27),  # left leg
            (24, 26), (26, 28),  # right leg
            (11, 12),            # shoulders
            (23, 24),            # hips
            (11, 23), (12, 24),  # torso
            (0, 11), (0, 12)     # head to shoulders
        ]

        for a, b in mp_connections:
            xa, ya, _ = landmarks_33[a]
            xb, yb, _ = landmarks_33[b]
            cv2.line(annotated, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

        return annotated

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
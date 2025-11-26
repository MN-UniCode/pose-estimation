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
    
    def draw_cv_barchart(self, width=640, height=480, max_value=200):
        graph = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas

        margin_left = 60
        margin_bottom = 50
        margin_top = 20

        bar_width = int((width - margin_left - 20) / len(self.group_names))
        
        # Draw axes
        cv2.line(graph, (margin_left, margin_top), (margin_left, height - margin_bottom), (0, 0, 0), 2)
        cv2.line(graph, (margin_left, height - margin_bottom), (width - 10, height - margin_bottom), (0, 0, 0), 2)

        # Extract last KE values for each group
        values = []
        for name in self.group_names:
            hist = self.ke_histories[f"{name}_ke"]
            values.append(hist[-1] if len(hist) > 0 else 0.0)

        # Plot bars
        for i, (name, value) in enumerate(zip(self.group_names, values)):
            bar_height = int((min(value, max_value) / max_value) * (height - margin_bottom - margin_top))
            x1 = margin_left + i * bar_width + 5
            y1 = height - margin_bottom - bar_height
            x2 = x1 + bar_width - 10
            y2 = height - margin_bottom

            cv2.rectangle(graph, (x1, y1), (x2, y2), (0, 122, 255), -1)

            # Label group name
            cv2.putText(graph, name.replace("_", ""),
                        (x1, height - margin_bottom + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

            # Label value
            cv2.putText(graph, f"{value:.1f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # Y-axis label
        cv2.putText(graph, "K.E.", (5, margin_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        return graph



    # Drawing a graph over time using OpenCV
    def draw_cv_graph(self, history, width=640, height=480, max_value=2.0, fps=25, window_length=5, y_label="y"):
        graph = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

        # Axes
        cv2.line(graph, (50, 0), (50, height - 40), (0, 0, 0), 1)  # y-axis
        cv2.line(graph, (50, height - 40), (width, height - 40), (0, 0, 0), 1)  # x-axis

        # Y-axis labels (fixed range)
        for i in range(5):
            y_value = max_value * i / 4
            y_pos = int(height - 40 - (y_value / max_value) * (height - 50))
            label = f"{y_value:.1f}"
            cv2.putText(graph, label, (5, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # X-axis time ticks
        history_length = fps * window_length
        seconds_range = history_length / fps
        tick_px = (width - 50) / seconds_range

        for i in range(int(seconds_range) + 1):
            x = int(50 + i * tick_px)
            cv2.line(graph, (x, height - 40), (x, height - 35), (0, 0, 0), 1)
            cv2.putText(graph, f"{i}s", (x - 10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Plot line
        if len(history) >= 2:
            for i in range(1, len(history)):
                x1 = int(50 + (i - 1) / history_length * (width - 50))
                x2 = int(50 + i / history_length * (width - 50))

                y1 = int(height - 40 - (min(history[i - 1], max_value) / max_value) * (height - 50))
                y2 = int(height - 40 - (min(history[i], max_value) / max_value) * (height - 50))

                cv2.line(graph, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Axis labels
        cv2.putText(graph, y_label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        cv2.putText(graph, "Time (s)", (width // 2, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        return graph


    # Stack OpenCV images horizontally
    def stack_images_horizontal(self, images, scale=1.0):
        resized_images = []
        for img in images:
            if len(img.shape) == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            resized_images.append(img)
        return cv2.hconcat(resized_images)
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
    
    def draw_cv_barchart(self, ke, group_names, width=640, height=480, max_value=200):
        graph = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas

        margin_left = 100
        margin_bottom = 60
        margin_top = 20

        font_scale = 1.5
        thickness = 2

        bar_width = int((width - margin_left - 20) / len(group_names))
        
        # Draw axes
        cv2.line(graph, (margin_left, margin_top), (margin_left, height - margin_bottom), (0, 0, 0), 2)
        cv2.line(graph, (margin_left, height - margin_bottom), (width - 10, height - margin_bottom), (0, 0, 0), 2)

        # Extract last KE values for each group
        values = []
        for name in group_names:
            hist = ke[f"{name}_ke"]
            values.append(hist)

        # Plot bars
        for i, (name, value) in enumerate(zip(group_names, values)):
            bar_height = int((min(value, max_value) / max_value) * (height - margin_bottom - margin_top))
            x1 = margin_left + i * bar_width + 5
            y1 = height - margin_bottom - bar_height
            x2 = x1 + bar_width - 10
            y2 = height - margin_bottom

            cv2.rectangle(graph, (x1, y1), (x2, y2), (0, 122, 255), -1)

            # Label group name
            cv2.putText(graph, name.replace("_", ""),
                        (x1, height - margin_bottom + 47),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

            # Label value
            cv2.putText(graph, f"{value:.1f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

        # Y-axis label
        cv2.putText(graph, "K.E.", (5, margin_top + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), thickness)

        return graph

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
    def stack_images_horizontal(self, images, scale=1.0):
        resized_images = []
        for img in images:
            if len(img.shape) == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            resized_images.append(img)
        return cv2.hconcat(resized_images)
import cv2
from config_manager import ConfigManager
from typing import List, Dict, Any, Optional
import numpy as np

class DisplayManager:
    """Manages drawing detections, stats, and UI elements on frames."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.show_fps = self.config.display_config.get('show_fps', True)
        self.show_confidence = self.config.display_config.get('show_confidence', True)
        self.show_stats = self.config.display_config.get('show_stats', True)
        self.box_thickness = self.config.display_config.get('box_thickness', 2)
        self.font_scale = self.config.display_config.get('font_scale', 0.6)
        self.window_name = self.config.display_config.get('window_name', "Object Detector - YOLOv8")
        self.class_names = self.config.class_names
        
        # Initialize stats
        self.detection_stats = {class_id: 0 for class_id in self.class_names.keys()}

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draws bounding boxes and labels for detections."""
        overlay = frame.copy()
        # Reset stats for this frame
        self.detection_stats = {class_id: 0 for class_id in self.class_names.keys()}
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']

            # Update stats
            self.detection_stats[class_id] = self.detection_stats.get(class_id, 0) + 1

            # Get color for class using the config manager
            color = self.config.get_color_for_class(class_id)
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Prepare label
            if self.show_confidence:
                label = f"{class_name}: {confidence*100:.0f}%"
            else:
                label = class_name

            # Get text size and draw background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
            )
            cv2.rectangle(
                overlay, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1
            )
            # Draw text
            cv2.putText(
                overlay, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.config.get_ui_color('text_fg'), 2
            )
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame

    def draw_stats(self, frame: np.ndarray, fps: float, is_paused: bool = False) -> np.ndarray:
        """Draws FPS, object counts, and pause indicator."""
        h, w = frame.shape[:2]

        if self.show_stats:
            stats_height = 120
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (220, stats_height), self.config.get_ui_color('stats_bg'), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            y_offset = 35
            if self.show_fps:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.get_ui_color('text_fg'), 2)
                y_offset += 25

            total_objects = sum(self.detection_stats.values())
            cv2.putText(frame, f"Objects: {total_objects}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.get_ui_color('text_fg'), 2)
            y_offset += 25

            # Display details by class, using their specific colors
            for class_id, count in self.detection_stats.items():
                if count > 0:
                    name = self.class_names.get(class_id, 'Unknown')
                    color = self.config.get_color_for_class(class_id) # Get class-specific color
                    cv2.putText(frame, f"  {name}: {count}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y_offset += 20

        if is_paused:
            pause_text = "PAUSE"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            x = (w - text_size[0]) // 2
            y = h // 2
            cv2.rectangle(frame, (x - 20, y - 40), (x + text_size[0] + 20, y + 20),
                         self.config.get_ui_color('text_bg'), -1)
            cv2.putText(frame, pause_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3) # Yellow color for pause

        help_text = "SPACE: Pause | S: Screenshot | Q: Exit"
        cv2.putText(frame, help_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.get_ui_color('text_fg'), 1)
        
        return frame

    def show_frame(self, frame: np.ndarray):
        """Displays the frame in the named window."""
        cv2.imshow(self.window_name, frame)

    def create_window(self):
        """Creates the display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
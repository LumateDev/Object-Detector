import cv2
from config_manager import ConfigManager
from typing import Optional
import numpy as np

class CameraManager:
    """Manages webcam initialization and frame capture."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._initialize_camera()

    def _initialize_camera(self) -> bool:
        """Initializes the webcam."""
        camera_idx = self.config.camera_config.get('index', 0)
        width = self.config.camera_config.get('width', 640)
        height = self.config.camera_config.get('height', 480)
        fps = self.config.camera_config.get('fps', 30)

        print(f"[INFO] Connecting to camera {camera_idx}...")
        self.cap = cv2.VideoCapture(camera_idx)

        if not self.cap.isOpened():
            print(f"[INFO] Camera {camera_idx} failed, trying others...")
            for idx in range(3):
                if idx != camera_idx:
                    self.cap = cv2.VideoCapture(idx)
                    if self.cap.isOpened():
                        print(f"[INFO] Camera found at index {idx}")
                        break

        if not self.cap.isOpened():
            print("[ERROR] ❌ Webcam not found!")
            return False

        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"[INFO] ✅ Camera connected: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads a frame from the camera."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Mirror flip for convenience
                return cv2.flip(frame, 1)
        print("[WARN] Failed to get frame from camera")
        return None

    def release(self):
        """Releases the camera resource."""
        if self.cap:
            self.cap.release()
            self.cap = None
from ultralytics import YOLO
from config_manager import ConfigManager
from typing import List, Dict, Any
import numpy as np

class ModelManager:
    """Manages YOLO model loading and inference."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_name = self.config.detection_config.get('model', 'yolov8n.pt')
        self.confidence_threshold = self.config.detection_config.get('confidence', 0.5)
        self.target_classes = self.config.detection_config.get('classes', [0, 41, 67])
        self.model: YOLO = self._load_model()

    def _load_model(self) -> YOLO:
        """Loads the YOLO model."""
        print(f"[INFO] Loading model {self.model_name}...")
        try:
            model = YOLO(self.model_name)
            print(f"[INFO] ✅ Model {self.model_name} loaded.")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load model {self.model_name}: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Performs object detection on a single frame."""
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=self.target_classes,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id,
                    'class_name': self.config.class_names.get(class_id, 'Unknown'),
                    'confidence': confidence
                })
        return detections
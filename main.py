#!/usr/bin/env python3
"""
Object Detector v1.0
Object detection (person, cup, phone) in real-time
using webcam and YOLOv8
"""

import cv2
import numpy as np
import time
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional, Dict, List, Tuple, Any

# Check and install dependencies
def check_dependencies():
    """Check for required libraries"""
    missing = []
    
    try:
        import ultralytics
    except ImportError:
        missing.append('ultralytics')
    
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    
    try:
        import yaml
    except ImportError:
        missing.append('PyYAML')
    
    if missing:
        print(f"❌ Missing libraries: {', '.join(missing)}")
        print(f"   Install with command: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from ultralytics import YOLO
import yaml


class Colors:
    """Color scheme for object classes"""
    # BGR format for OpenCV
    PERSON = (0, 255, 0)      # Green
    CUP = (255, 150, 0)       # Blue
    PHONE = (0, 165, 255)     # Orange
    
    # UI colors
    TEXT_BG = (0, 0, 0)       # Black background
    TEXT_FG = (255, 255, 255) # White text
    STATS_BG = (40, 40, 40)   # Dark gray
    
    @classmethod
    def get_color(cls, class_id: int) -> Tuple[int, int, int]:
        """Get color by class ID"""
        color_map = {
            0: cls.PERSON,   # person
            41: cls.CUP,     # cup
            67: cls.PHONE    # cell phone
        }
        return color_map.get(class_id, (128, 128, 128))


class FPSCounter:
    """FPS counter with averaging"""
    
    def __init__(self, avg_frames: int = 30):
        self.times = deque(maxlen=avg_frames)
        self.last_time = time.time()
    
    def update(self) -> float:
        """Update and get current FPS"""
        current_time = time.time()
        self.times.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.times) > 0:
            return 1.0 / (sum(self.times) / len(self.times))
        return 0.0


class Config:
    """Configuration manager"""
    
    DEFAULT_CONFIG = {
        'camera': {
            'index': 0,
            'width': 640,
            'height': 480,
            'fps': 30
        },
        'detection': {
            'model': 'yolov8n.pt',
            'confidence': 0.5,
            'classes': [0, 41, 67]
        },
        'display': {
            'show_fps': True,
            'show_confidence': True,
            'show_stats': True,
            'box_thickness': 2,
            'font_scale': 0.6
        },
        'mode': 'balanced'
    }
    
    MODES = {
        'fast': {
            'camera': {'width': 320, 'height': 240},
            'detection': {'confidence': 0.4}
        },
        'balanced': {
            'camera': {'width': 640, 'height': 480},
            'detection': {'confidence': 0.5}
        },
        'accurate': {
            'camera': {'width': 1280, 'height': 720},
            'detection': {'confidence': 0.6}
        }
    }
    
    def __init__(self, config_path: Optional[str] = None, mode: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Apply mode
        if mode and mode in self.MODES:
            self._apply_mode(mode)
    
    def _load_from_file(self, path: str):
        """Load configuration from YAML file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self._deep_update(self.config, loaded)
            print(f"[INFO] Configuration loaded from {path}")
        except Exception as e:
            print(f"[WARN] Config load error: {e}")
    
    def _deep_update(self, base: dict, update: dict):
        """Recursive dictionary update"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _apply_mode(self, mode: str):
        """Apply preset mode"""
        if mode in self.MODES:
            self._deep_update(self.config, self.MODES[mode])
            print(f"[INFO] Applied mode: {mode}")
    
    def get(self, *keys):
        """Get value by keys"""
        value = self.config
        for key in keys:
            value = value.get(key, {})
        return value


class ObjectDetector:
    """Main object detector class"""
    
    # Class names in English
    CLASS_NAMES = {
        0: 'Person',
        41: 'Cup',
        67: 'Phone'
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps_counter = FPSCounter()
        
        # Application state
        self.is_running = True
        self.is_paused = False
        self.show_fps = config.get('display', 'show_fps')
        self.show_confidence = config.get('display', 'show_confidence')
        self.show_stats = config.get('display', 'show_stats')
        
        # Statistics
        self.detection_stats: Dict[int, int] = {0: 0, 41: 0, 67: 0}
        self.last_frame: Optional[np.ndarray] = None
        
        # Screenshots directory
        self.screenshots_dir = Path('screenshots')
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Display parameters
        self.box_thickness = config.get('display', 'box_thickness')
        self.font_scale = config.get('display', 'font_scale')
        self.target_classes = config.get('detection', 'classes')
        self.confidence_threshold = config.get('detection', 'confidence')
    
    def print_banner(self):
        """Print welcome banner"""
        banner = """
╔═══════════════════════════════════════════╗
║       🎯 Object Detector v1.0             ║
║       For laptop with webcam              ║
╠═══════════════════════════════════════════╣
║  Detected objects:                        ║
║    🟢 Person (person)                     ║
║    🔵 Cup (cup)                           ║
║    🟠 Phone (cell phone)                  ║
╠═══════════════════════════════════════════╣
║  Controls:                                ║
║    SPACE   - Start/Pause detection        ║
║    S       - Save screenshot              ║
║    F       - Show/hide FPS                ║
║    C       - Show/hide confidence         ║
║    I       - Show/hide statistics         ║
║    Q / ESC - Exit                         ║
╚═══════════════════════════════════════════╝
"""
        print(banner)
    
    def initialize(self) -> bool:
        """Initialize all components"""
        print("\n[INFO] Initialization...")
        
        # 1. Check and load model
        if not self._load_model():
            return False
        
        # 2. Initialize camera
        if not self._init_camera():
            return False
        
        # 3. Detect resources
        self._detect_resources()
        
        print("[INFO] ✅ Initialization completed successfully!\n")
        return True
    
    def _load_model(self) -> bool:
        """Load YOLO model"""
        model_name = self.config.get('detection', 'model')
        print(f"[INFO] Loading model {model_name}...")
        
        try:
            # Progress bar (emulation)
            self._print_progress("Loading model", 0)
            
            self.model = YOLO(model_name)
            
            self._print_progress("Loading model", 100)
            print()
            
            # Model info
            model_path = Path(model_name)
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"[INFO] Model loaded ({size_mb:.1f} MB)")
            else:
                print(f"[INFO] Model loaded (downloaded automatically)")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Model load error: {e}")
            return False
    
    def _init_camera(self) -> bool:
        """Initialize webcam"""
        camera_index = self.config.get('camera', 'index')
        width = self.config.get('camera', 'width')
        height = self.config.get('camera', 'height')
        fps = self.config.get('camera', 'fps')
        
        print(f"[INFO] Connecting to camera {camera_index}...")
        
        # Try to connect to camera
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            # Try other indices
            for idx in range(3):
                if idx != camera_index:
                    self.cap = cv2.VideoCapture(idx)
                    if self.cap.isOpened():
                        print(f"[INFO] Camera found at index {idx}")
                        break
        
        if not self.cap.isOpened():
            print("[ERROR] ❌ Webcam not found!")
            print("        Check camera connection")
            return False
        
        # Set camera parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Get actual parameters
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[INFO] ✅ Camera connected: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        return True
    
    def _detect_resources(self):
        """Detect available resources"""
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] GPU detected: {gpu_name}")
            print("[INFO] Mode: CUDA (GPU)")
        else:
            print("[INFO] GPU not detected")
            print("[INFO] Mode: CPU (expected FPS: 10-20)")
    
    def _print_progress(self, label: str, percent: int):
        """Print progress bar"""
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[INFO] {label}: [{bar}] {percent}%", end='', flush=True)
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Perform object detection"""
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
                
                # Coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id,
                    'class_name': self.CLASS_NAMES.get(class_id, 'Unknown'),
                    'confidence': confidence
                })
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results"""
        overlay = frame.copy()
        
        # Reset statistics
        self.detection_stats = {0: 0, 41: 0, 67: 0}
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Update statistics
            self.detection_stats[class_id] = self.detection_stats.get(class_id, 0) + 1
            
            # Color for class
            color = Colors.get_color(class_id)
            
            # Bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Label
            if self.show_confidence:
                label = f"{class_name}: {confidence*100:.0f}%"
            else:
                label = class_name
            
            # Text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
            )
            
            # Text background
            cv2.rectangle(
                overlay,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                overlay,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                Colors.TEXT_FG,
                2
            )
        
        # Blend with original for transparency
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        return frame
    
    def draw_stats(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw statistics"""
        h, w = frame.shape[:2]
        
        # Stats panel
        if self.show_stats:
            stats_height = 120
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (220, stats_height), Colors.STATS_BG, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            y_offset = 35
            
            # FPS
            if self.show_fps:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.TEXT_FG, 2)
                y_offset += 25
            
            # Object count
            total_objects = sum(self.detection_stats.values())
            cv2.putText(frame, f"Objects: {total_objects}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.TEXT_FG, 2)
            y_offset += 25
            
            # Details by class
            for class_id, count in self.detection_stats.items():
                if count > 0:
                    name = self.CLASS_NAMES[class_id]
                    color = Colors.get_color(class_id)
                    cv2.putText(frame, f"  {name}: {count}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y_offset += 20
        
        # Pause indicator
        if self.is_paused:
            pause_text = "PAUSE"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            x = (w - text_size[0]) // 2
            y = h // 2
            
            # Background
            cv2.rectangle(frame, (x - 20, y - 40), (x + text_size[0] + 20, y + 20),
                         Colors.TEXT_BG, -1)
            cv2.putText(frame, pause_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Control hints at bottom
        help_text = "SPACE: Pause | S: Screenshot | Q: Exit"
        cv2.putText(frame, help_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_FG, 1)
        
        return frame
    
    def save_screenshot(self, frame: np.ndarray):
        """Save screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshots_dir / f"detection_{timestamp}.jpg"
        
        cv2.imwrite(str(filename), frame)
        print(f"[INFO] 📷 Screenshot saved: {filename}")
    
    def handle_key(self, key: int) -> bool:
        """Handle key presses"""
        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            print("\n[INFO] Exiting program...")
            return False
        
        elif key == ord(' '):  # Space - pause
            self.is_paused = not self.is_paused
            status = "⏸ Pause" if self.is_paused else "▶ Resume"
            print(f"[INFO] {status}")
        
        elif key == ord('s') or key == ord('S'):  # Screenshot
            if self.last_frame is not None:
                self.save_screenshot(self.last_frame)
        
        elif key == ord('f') or key == ord('F'):  # Toggle FPS
            self.show_fps = not self.show_fps
            print(f"[INFO] FPS: {'shown' if self.show_fps else 'hidden'}")
        
        elif key == ord('c') or key == ord('C'):  # Toggle confidence
            self.show_confidence = not self.show_confidence
            print(f"[INFO] Confidence: {'shown' if self.show_confidence else 'hidden'}")
        
        elif key == ord('i') or key == ord('I'):  # Toggle statistics
            self.show_stats = not self.show_stats
            print(f"[INFO] Statistics: {'shown' if self.show_stats else 'hidden'}")
        
        return True
    
    def run(self):
        """Main execution loop"""
        self.print_banner()
        
        if not self.initialize():
            print("\n[ERROR] Initialization failed. Exiting.")
            return
        
        print("[INFO] ▶ Detection started!")
        print("[INFO] Use hotkeys for control\n")
        
        window_name = "Object Detector - YOLOv8"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("[WARN] Failed to get frame from camera")
                    continue
                
                # Mirror flip for convenience
                frame = cv2.flip(frame, 1)
                
                # Detection (if not paused)
                if not self.is_paused:
                    detections = self.detect(frame)
                    frame = self.draw_detections(frame, detections)
                else:
                    # On pause use last detections
                    detections = []
                
                # Update FPS
                fps = self.fps_counter.update()
                
                # Draw statistics
                frame = self.draw_stats(frame, fps)
                
                # Save last frame for screenshot
                self.last_frame = frame.copy()
                
                # Display
                cv2.imshow(window_name, frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl+C)")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        print("[INFO] Releasing resources...")
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("[INFO] ✅ Done. Goodbye!")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Object Detector - Object detection with webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python main.py                    # Run with default settings
  python main.py --fast             # Fast mode (low quality)
  python main.py --accurate         # Accurate mode (high quality)
  python main.py --config my.yaml   # Use custom config
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--fast', action='store_true',
                           help='Fast mode (320x240, low threshold)')
    mode_group.add_argument('--balanced', action='store_true',
                           help='Balanced mode (default)')
    mode_group.add_argument('--accurate', action='store_true',
                           help='Accurate mode (1280x720, high threshold)')
    
    parser.add_argument('--camera', '-cam', type=int, default=None,
                       help='Camera index (default: 0)')
    
    parser.add_argument('--confidence', '-conf', type=float, default=None,
                       help='Confidence threshold (0.0-1.0)')
    
    return parser.parse_args()


def main():
    """Entry point"""
    args = parse_arguments()
    
    # Determine mode
    mode = None
    if args.fast:
        mode = 'fast'
    elif args.accurate:
        mode = 'accurate'
    elif args.balanced:
        mode = 'balanced'
    
    # Load configuration
    config = Config(args.config, mode)
    
    # Override from command line arguments
    if args.camera is not None:
        config.config['camera']['index'] = args.camera
    
    if args.confidence is not None:
        config.config['detection']['confidence'] = args.confidence
    
    # Start detector
    detector = ObjectDetector(config)
    detector.run()


if __name__ == "__main__":
    main()
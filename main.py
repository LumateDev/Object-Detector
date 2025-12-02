import argparse
from pathlib import Path
import cv2
from config_manager import ConfigManager
from camera_manager import CameraManager
from model_manager import ModelManager
from display_manager import DisplayManager
from input_handler import InputHandler
from utils import FPSCounter

class ObjectDetectorApp:
    """Main application class orchestrating all components."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.camera = CameraManager(config)
        self.model = ModelManager(config)
        self.display = DisplayManager(config)
        self.input_handler = InputHandler()
        self.fps_counter = FPSCounter()
        
        self.screenshots_dir = self.config.screenshots_dir
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Share state between input and display
        self.display.show_fps = self.input_handler.show_fps
        self.display.show_confidence = self.input_handler.show_confidence
        self.display.show_stats = self.input_handler.show_stats

    def print_banner(self):
        """Prints the welcome banner."""
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

    def run(self):
        """Main application loop."""
        self.print_banner()
        
        if not self.camera.cap:
            print("\n[ERROR] Camera initialization failed. Cannot run.")
            return

        self.display.create_window()
        last_frame = None
        
        print("[INFO] ▶ Detection started!")
        print("[INFO] Use hotkeys for control\n")

        try:
            while self.input_handler.is_running:
                frame = self.camera.read_frame()
                if frame is None:
                    continue

                last_frame = frame.copy() # For screenshots

                # Update display settings from input handler
                self.display.show_fps = self.input_handler.show_fps
                self.display.show_confidence = self.input_handler.show_confidence
                self.display.show_stats = self.input_handler.show_stats

                if not self.input_handler.is_paused:
                    detections = self.model.detect(frame)
                    frame = self.display.draw_detections(frame, detections)
                else:
                    detections = [] # Or use last detections if stored elsewhere

                fps = self.fps_counter.update()
                frame = self.display.draw_stats(frame, fps, self.input_handler.is_paused)
                
                self.display.show_frame(frame)
                
                key = self.input_handler.get_key(1)
                action = self.input_handler.handle_key(key)
                
                if action == 'screenshot' and last_frame is not None:
                    self._save_screenshot(last_frame)
                
                if action is False: # Exit signal
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl+C)")
        finally:
            self.cleanup()

    def _save_screenshot(self, frame):
        """Saves the current frame as a screenshot."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshots_dir / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"[INFO] 📷 Screenshot saved: {filename}")

    def cleanup(self):
        """Releases resources."""
        print("[INFO] Releasing resources...")
        self.camera.release()
        cv2.destroyAllWindows()
        print("[INFO] ✅ Done. Goodbye!")

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Object Detector - Object detection with webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python main.py                    # Run with default settings
  python main.py --fast             # Fast mode (low quality)
  python main.py --balanced         # Balanced mode (default)
  python main.py --accurate         # Accurate mode (high quality)
  python main.py --very-accurate    # Very accurate mode (higher res, better model)
  python main.py --ultra-accurate   # Ultra accurate mode (highest res, best model, high conf)
  python main.py --high-res-accurate # High resolution mode
  python main.py --high-conf-accurate # High confidence mode
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
                           help='Accurate mode (1280x720, higher conf)')
    mode_group.add_argument('--very-accurate', action='store_true',
                           help='Very accurate mode (1280x720, yolo-m, higher conf)')
    mode_group.add_argument('--ultra-accurate', action='store_true',
                           help='Ultra accurate mode (1920x1080, yolo-x, high conf)')
    mode_group.add_argument('--high-res-accurate', action='store_true',
                           help='High resolution accurate mode (1920x1080, default model/conf)')
    mode_group.add_argument('--high-conf-accurate', action='store_true',
                           help='High confidence accurate mode (high conf, default res/model)')

    parser.add_argument('--camera', '-cam', type=int, default=None,
                       help='Camera index (default: 0)')

    parser.add_argument('--confidence', '-conf', type=float, default=None,
                       help='Confidence threshold (0.0-1.0)')

    return parser.parse_args()

def main():
    """Entry point."""
    args = parse_arguments()
    
    mode = None
    if args.fast:
        mode = 'fast'
    elif args.balanced:
        mode = 'balanced'
    elif args.accurate:
        mode = 'accurate'
    elif args.very_accurate:
        mode = 'very_accurate'
    elif args.ultra_accurate:
        mode = 'ultra_accurate'
    elif args.high_res_accurate:
        mode = 'high_res_accurate'
    elif args.high_conf_accurate:
        mode = 'high_conf_accurate'
    # 'balanced' is default if no mode is specified, handled in ConfigManager

    # Load configuration
    config = ConfigManager(args.config, mode)

    # Override from command line arguments if provided
    if args.camera is not None:
        config.config['camera']['index'] = args.camera
    if args.confidence is not None:
        config.config['detection']['confidence'] = args.confidence

    # Start the application
    app = ObjectDetectorApp(config)
    app.run()

if __name__ == "__main__":
    main()
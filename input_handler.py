import cv2
from typing import Tuple
import numpy as np

class InputHandler:
    """Handles keyboard input and user actions."""
    
    def __init__(self):
        self.is_running = True
        self.is_paused = False
        self.show_fps = True
        self.show_confidence = True
        self.show_stats = True

    def handle_key(self, key_code: int) -> bool: # Returns True if continue running
        """Processes a key press and updates application state."""
        key = key_code & 0xFF  # Mask to get ASCII value

        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            print("\n[INFO] Exiting program...")
            self.is_running = False
            return False
        
        elif key == ord(' '):  # Space - pause
            self.is_paused = not self.is_paused
            status = "⏸ Pause" if self.is_paused else "▶ Resume"
            print(f"[INFO] {status}")
        
        elif key == ord('s') or key == ord('S'):  # Screenshot
            return 'screenshot' # Signal to main app to take screenshot
        
        elif key == ord('f') or key == ord('F'):  # Toggle FPS
            self.show_fps = not self.show_fps
            print(f"[INFO] FPS: {'shown' if self.show_fps else 'hidden'}")
        
        elif key == ord('c') or key == ord('C'):  # Toggle confidence
            self.show_confidence = not self.show_confidence
            print(f"[INFO] Confidence: {'shown' if self.show_confidence else 'hidden'}")
        
        elif key == ord('i') or key == ord('I'):  # Toggle statistics
            self.show_stats = not self.show_stats
            print(f"[INFO] Statistics: {'shown' if self.show_stats else 'hidden'}")

        return True # Continue running

    def get_key(self, delay: int = 1) -> int:
        """Gets the pressed key code."""
        return cv2.waitKey(delay)
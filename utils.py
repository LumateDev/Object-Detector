from collections import deque
import time
from typing import Tuple

class FPSCounter:
    """FPS counter with averaging."""
    def __init__(self, avg_frames: int = 30):
        self.times = deque(maxlen=avg_frames)
        self.last_time = time.time()

    def update(self) -> float:
        """Update and get current FPS."""
        current_time = time.time()
        self.times.append(current_time - self.last_time)
        self.last_time = current_time

        if len(self.times) > 0:
            return 1.0 / (sum(self.times) / len(self.times))
        return 0.0

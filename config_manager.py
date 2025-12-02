import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages application configuration from a YAML file."""

    DEFAULT_CONFIG_PATH = 'config.yaml'
    # Убираем DEFAULT_CONFIG из кода, полагаясь на config.yaml
    # DEFAULT_CONFIG = {...}

    def __init__(self, config_path: Optional[str] = None, mode: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._get_default_config() # Загружаем дефолт из кода
        self._load_config()
        if mode:
            self._apply_mode(mode)

    def _get_default_config(self) -> Dict[str, Any]:
        """Returns the default configuration structure."""
        return {
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
                'font_scale': 0.6,
                'window_name': "Object Detector - YOLOv8"
            },
            'mode_presets': {
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
            },
            'screenshots_dir': 'screenshots',
            'class_colors': {
                0: (0, 255, 0),    # Person - Green
                41: (255, 150, 0), # Cup - Blue
                67: (0, 165, 255), # Phone - Orange
            },
            'ui_colors': {
                'text_bg': (0, 0, 0),       # Black
                'text_fg': (255, 255, 255), # White
                'stats_bg': (40, 40, 40),   # Dark Gray
            },
            'class_names': {
                0: 'Person',
                41: 'Cup',
                67: 'Phone'
            }
        }


    def _load_config(self):
        """Loads configuration from the YAML file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        self._deep_update(self.config, file_config)
                print(f"[INFO] Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"[WARN] Could not load config from {self.config_path}: {e}. Using defaults.")
        else:
            print(f"[INFO] Config file {self.config_path} not found. Using defaults.")

    def _apply_mode(self, mode: str):
        """Applies settings from a preset mode."""
        if mode in self.config.get('mode_presets', {}):
            mode_config = self.config['mode_presets'][mode]
            self._deep_update(self.config, mode_config)
            print(f"[INFO] Applied mode: {mode}")
        else:
            print(f"[WARN] Mode '{mode}' not found in presets. Using defaults.")

    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Recursively updates nested dictionaries."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def get(self, *keys: str) -> Any:
        """Gets a configuration value by nested keys."""
        value = self.config
        for key in keys:
            value = value.get(key, {})
        return value

    @property
    def camera_config(self) -> Dict[str, Any]:
        return self.config.get('camera', {})

    @property
    def detection_config(self) -> Dict[str, Any]:
        return self.config.get('detection', {})

    @property
    def display_config(self) -> Dict[str, Any]:
        return self.config.get('display', {})

    @property
    def class_colors(self) -> Dict[int, tuple]:
        return self.config.get('class_colors', {})

    @property
    def ui_colors(self) -> Dict[str, tuple]:
        return self.config.get('ui_colors', {})

    @property
    def class_names(self) -> Dict[int, str]:
        return self.config.get('class_names', {})

    @property
    def screenshots_dir(self) -> Path:
        return Path(self.config.get('screenshots_dir', 'screenshots'))

    def get_color_for_class(self, class_id: int) -> tuple:
        """Returns the color for a given class ID."""
        color = self.class_colors.get(class_id)
        if color is not None:
            return color
        return (128, 128, 128)

    def get_ui_color(self, name: str) -> tuple:
        """Returns the UI color by name."""
        return self.ui_colors.get(name, (255, 255, 255)) # White default
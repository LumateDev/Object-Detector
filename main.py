#!/usr/bin/env python3
"""
Object Detector v1.0
Детекция объектов (человек, чашка, телефон) в реальном времени
с использованием веб-камеры и YOLOv8
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

# Проверка и установка зависимостей
def check_dependencies():
    """Проверка наличия необходимых библиотек"""
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
        print(f"❌ Отсутствуют библиотеки: {', '.join(missing)}")
        print(f"   Установите их командой: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from ultralytics import YOLO
import yaml


class Colors:
    """Цветовая схема для классов объектов"""
    # BGR формат для OpenCV
    PERSON = (0, 255, 0)      # Зелёный
    CUP = (255, 150, 0)       # Синий
    PHONE = (0, 165, 255)     # Оранжевый
    
    # Цвета интерфейса
    TEXT_BG = (0, 0, 0)       # Чёрный фон
    TEXT_FG = (255, 255, 255) # Белый текст
    STATS_BG = (40, 40, 40)   # Тёмно-серый
    
    @classmethod
    def get_color(cls, class_id: int) -> Tuple[int, int, int]:
        """Получить цвет по ID класса"""
        color_map = {
            0: cls.PERSON,   # person
            41: cls.CUP,     # cup
            67: cls.PHONE    # cell phone
        }
        return color_map.get(class_id, (128, 128, 128))


class FPSCounter:
    """Счётчик FPS с усреднением"""
    
    def __init__(self, avg_frames: int = 30):
        self.times = deque(maxlen=avg_frames)
        self.last_time = time.time()
    
    def update(self) -> float:
        """Обновить и получить текущий FPS"""
        current_time = time.time()
        self.times.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.times) > 0:
            return 1.0 / (sum(self.times) / len(self.times))
        return 0.0


class Config:
    """Менеджер конфигурации"""
    
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
        
        # Загрузка из файла
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Применение режима
        if mode and mode in self.MODES:
            self._apply_mode(mode)
    
    def _load_from_file(self, path: str):
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self._deep_update(self.config, loaded)
            print(f"[INFO] Конфигурация загружена из {path}")
        except Exception as e:
            print(f"[WARN] Ошибка загрузки конфига: {e}")
    
    def _deep_update(self, base: dict, update: dict):
        """Рекурсивное обновление словаря"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _apply_mode(self, mode: str):
        """Применение предустановленного режима"""
        if mode in self.MODES:
            self._deep_update(self.config, self.MODES[mode])
            print(f"[INFO] Применён режим: {mode}")
    
    def get(self, *keys):
        """Получить значение по ключам"""
        value = self.config
        for key in keys:
            value = value.get(key, {})
        return value


class ObjectDetector:
    """Основной класс детектора объектов"""
    
    # Названия классов на русском
    CLASS_NAMES = {
        0: 'Человек',
        41: 'Чашка',
        67: 'Телефон'
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps_counter = FPSCounter()
        
        # Состояние приложения
        self.is_running = True
        self.is_paused = False
        self.show_fps = config.get('display', 'show_fps')
        self.show_confidence = config.get('display', 'show_confidence')
        self.show_stats = config.get('display', 'show_stats')
        
        # Статистика
        self.detection_stats: Dict[int, int] = {0: 0, 41: 0, 67: 0}
        self.last_frame: Optional[np.ndarray] = None
        
        # Папка для скриншотов
        self.screenshots_dir = Path('screenshots')
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Параметры отображения
        self.box_thickness = config.get('display', 'box_thickness')
        self.font_scale = config.get('display', 'font_scale')
        self.target_classes = config.get('detection', 'classes')
        self.confidence_threshold = config.get('detection', 'confidence')
    
    def print_banner(self):
        """Вывод приветственного баннера"""
        banner = """
╔═══════════════════════════════════════════╗
║       🎯 Object Detector v1.0             ║
║       Для ноутбука с веб-камерой          ║
╠═══════════════════════════════════════════╣
║  Детектируемые объекты:                   ║
║    🟢 Человек (person)                    ║
║    🔵 Чашка (cup)                         ║
║    🟠 Телефон (cell phone)                ║
╠═══════════════════════════════════════════╣
║  Управление:                              ║
║    ПРОБЕЛ  - Старт/Пауза детекции        ║
║    S       - Сохранить скриншот          ║
║    F       - Показать/скрыть FPS         ║
║    C       - Показать/скрыть confidence  ║
║    I       - Показать/скрыть статистику  ║
║    Q / ESC - Выход                       ║
╚═══════════════════════════════════════════╝
"""
        print(banner)
    
    def initialize(self) -> bool:
        """Инициализация всех компонентов"""
        print("\n[INFO] Инициализация...")
        
        # 1. Проверка и загрузка модели
        if not self._load_model():
            return False
        
        # 2. Инициализация камеры
        if not self._init_camera():
            return False
        
        # 3. Определение ресурсов
        self._detect_resources()
        
        print("[INFO] ✅ Инициализация завершена успешно!\n")
        return True
    
    def _load_model(self) -> bool:
        """Загрузка модели YOLO"""
        model_name = self.config.get('detection', 'model')
        print(f"[INFO] Загрузка модели {model_name}...")
        
        try:
            # Прогресс-бар (эмуляция)
            self._print_progress("Загрузка модели", 0)
            
            self.model = YOLO(model_name)
            
            self._print_progress("Загрузка модели", 100)
            print()
            
            # Информация о модели
            model_path = Path(model_name)
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"[INFO] Модель загружена ({size_mb:.1f} MB)")
            else:
                print(f"[INFO] Модель загружена (скачана автоматически)")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Ошибка загрузки модели: {e}")
            return False
    
    def _init_camera(self) -> bool:
        """Инициализация веб-камеры"""
        camera_index = self.config.get('camera', 'index')
        width = self.config.get('camera', 'width')
        height = self.config.get('camera', 'height')
        fps = self.config.get('camera', 'fps')
        
        print(f"[INFO] Подключение к камере {camera_index}...")
        
        # Попытка подключения к камере
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            # Попробуем другие индексы
            for idx in range(3):
                if idx != camera_index:
                    self.cap = cv2.VideoCapture(idx)
                    if self.cap.isOpened():
                        print(f"[INFO] Камера найдена на индексе {idx}")
                        break
        
        if not self.cap.isOpened():
            print("[ERROR] ❌ Веб-камера не найдена!")
            print("        Проверьте подключение камеры")
            return False
        
        # Настройка параметров камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Получение реальных параметров
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[INFO] ✅ Камера подключена: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        return True
    
    def _detect_resources(self):
        """Определение доступных ресурсов"""
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] GPU обнаружен: {gpu_name}")
            print("[INFO] Режим: CUDA (GPU)")
        else:
            print("[INFO] GPU не обнаружен")
            print("[INFO] Режим: CPU (ожидаемый FPS: 10-20)")
    
    def _print_progress(self, label: str, percent: int):
        """Вывод прогресс-бара"""
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[INFO] {label}: [{bar}] {percent}%", end='', flush=True)
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Выполнение детекции объектов"""
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
                
                # Координаты
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Класс и уверенность
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
        """Отрисовка результатов детекции"""
        overlay = frame.copy()
        
        # Сброс статистики
        self.detection_stats = {0: 0, 41: 0, 67: 0}
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Обновление статистики
            self.detection_stats[class_id] = self.detection_stats.get(class_id, 0) + 1
            
            # Цвет для класса
            color = Colors.get_color(class_id)
            
            # Bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Подпись
            if self.show_confidence:
                label = f"{class_name}: {confidence*100:.0f}%"
            else:
                label = class_name
            
            # Размер текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
            )
            
            # Фон для текста
            cv2.rectangle(
                overlay,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            
            # Текст
            cv2.putText(
                overlay,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                Colors.TEXT_FG,
                2
            )
        
        # Смешивание с оригиналом для полупрозрачности
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        return frame
    
    def draw_stats(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Отрисовка статистики"""
        h, w = frame.shape[:2]
        
        # Панель статистики
        if self.show_stats:
            stats_height = 120
            
            # Полупрозрачный фон
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
            
            # Количество объектов
            total_objects = sum(self.detection_stats.values())
            cv2.putText(frame, f"Объектов: {total_objects}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.TEXT_FG, 2)
            y_offset += 25
            
            # Детали по классам
            for class_id, count in self.detection_stats.items():
                if count > 0:
                    name = self.CLASS_NAMES[class_id]
                    color = Colors.get_color(class_id)
                    cv2.putText(frame, f"  {name}: {count}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y_offset += 20
        
        # Индикатор паузы
        if self.is_paused:
            pause_text = "⏸ ПАУЗА"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            x = (w - text_size[0]) // 2
            y = h // 2
            
            # Фон
            cv2.rectangle(frame, (x - 20, y - 40), (x + text_size[0] + 20, y + 20),
                         Colors.TEXT_BG, -1)
            cv2.putText(frame, pause_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Подсказка управления внизу
        help_text = "SPACE: Пауза | S: Скриншот | Q: Выход"
        cv2.putText(frame, help_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_FG, 1)
        
        return frame
    
    def save_screenshot(self, frame: np.ndarray):
        """Сохранение скриншота"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshots_dir / f"detection_{timestamp}.jpg"
        
        cv2.imwrite(str(filename), frame)
        print(f"[INFO] 📷 Скриншот сохранён: {filename}")
    
    def handle_key(self, key: int) -> bool:
        """Обработка нажатий клавиш"""
        if key == ord('q') or key == ord('Q') or key == 27:  # Q или ESC
            print("\n[INFO] Выход из программы...")
            return False
        
        elif key == ord(' '):  # Пробел - пауза
            self.is_paused = not self.is_paused
            status = "⏸ Пауза" if self.is_paused else "▶ Продолжение"
            print(f"[INFO] {status}")
        
        elif key == ord('s') or key == ord('S'):  # Скриншот
            if self.last_frame is not None:
                self.save_screenshot(self.last_frame)
        
        elif key == ord('f') or key == ord('F'):  # Переключение FPS
            self.show_fps = not self.show_fps
            print(f"[INFO] FPS: {'показан' if self.show_fps else 'скрыт'}")
        
        elif key == ord('c') or key == ord('C'):  # Переключение confidence
            self.show_confidence = not self.show_confidence
            print(f"[INFO] Confidence: {'показан' if self.show_confidence else 'скрыт'}")
        
        elif key == ord('i') or key == ord('I'):  # Переключение статистики
            self.show_stats = not self.show_stats
            print(f"[INFO] Статистика: {'показана' if self.show_stats else 'скрыта'}")
        
        return True
    
    def run(self):
        """Основной цикл работы"""
        self.print_banner()
        
        if not self.initialize():
            print("\n[ERROR] Инициализация не удалась. Выход.")
            return
        
        print("[INFO] ▶ Детекция запущена!")
        print("[INFO] Для управления используйте горячие клавиши\n")
        
        window_name = "Object Detector - YOLOv8"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self.is_running:
                # Чтение кадра
                ret, frame = self.cap.read()
                
                if not ret:
                    print("[WARN] Не удалось получить кадр с камеры")
                    continue
                
                # Зеркальное отражение для удобства
                frame = cv2.flip(frame, 1)
                
                # Детекция (если не на паузе)
                if not self.is_paused:
                    detections = self.detect(frame)
                    frame = self.draw_detections(frame, detections)
                else:
                    # На паузе используем последние детекции
                    detections = []
                
                # Обновление FPS
                fps = self.fps_counter.update()
                
                # Отрисовка статистики
                frame = self.draw_stats(frame, fps)
                
                # Сохранение последнего кадра для скриншота
                self.last_frame = frame.copy()
                
                # Отображение
                cv2.imshow(window_name, frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
        
        except KeyboardInterrupt:
            print("\n[INFO] Прервано пользователем (Ctrl+C)")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Освобождение ресурсов"""
        print("[INFO] Освобождение ресурсов...")
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("[INFO] ✅ Готово. До свидания!")


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Object Detector - Детекция объектов с веб-камеры",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py                    # Запуск с настройками по умолчанию
  python main.py --fast             # Быстрый режим (низкое качество)
  python main.py --accurate         # Точный режим (высокое качество)
  python main.py --config my.yaml   # Использовать свой конфиг
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации (по умолчанию: config.yaml)')
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--fast', action='store_true',
                           help='Быстрый режим (320x240, низкий порог)')
    mode_group.add_argument('--balanced', action='store_true',
                           help='Сбалансированный режим (по умолчанию)')
    mode_group.add_argument('--accurate', action='store_true',
                           help='Точный режим (1280x720, высокий порог)')
    
    parser.add_argument('--camera', '-cam', type=int, default=None,
                       help='Индекс камеры (по умолчанию: 0)')
    
    parser.add_argument('--confidence', '-conf', type=float, default=None,
                       help='Порог уверенности (0.0-1.0)')
    
    return parser.parse_args()


def main():
    """Точка входа"""
    args = parse_arguments()
    
    # Определение режима
    mode = None
    if args.fast:
        mode = 'fast'
    elif args.accurate:
        mode = 'accurate'
    elif args.balanced:
        mode = 'balanced'
    
    # Загрузка конфигурации
    config = Config(args.config, mode)
    
    # Переопределение из аргументов командной строки
    if args.camera is not None:
        config.config['camera']['index'] = args.camera
    
    if args.confidence is not None:
        config.config['detection']['confidence'] = args.confidence
    
    # Запуск детектора
    detector = ObjectDetector(config)
    detector.run()


if __name__ == "__main__":
    main()
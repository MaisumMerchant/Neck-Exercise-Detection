"""
Professional Head Pose Estimation and Neck Exercise Tracker
Using Kivy for real-time camera and UI
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from enum import Enum

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.clock import Clock
from kivy.core.window import Window


# ==============================
# Enums and Data Classes
# ==============================


class Direction(Enum):
    """Enum for head pose directions"""

    CENTER = "Center"
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    NOT_DETECTED = "Not Detected"


@dataclass
class ThresholdConfig:
    """Configuration for pose detection thresholds"""

    pitch_up: float = -35.0
    pitch_down: float = 35.0
    yaw_left: float = -45.0
    yaw_right: float = 45.0
    center_pitch: float = 15.0
    center_yaw: float = 10.0
    hold_required: int = 30
    min_time_between_reps: float = 0.8
    smoothing_alpha: float = 0.8


@dataclass
class RepCounter:
    """Tracks repetitions for paired movements (up-down and left-right)"""

    vertical: int = 0  # Counts up-down pairs
    horizontal: int = 0  # Counts left-right pairs
    last_vertical: Optional[Direction] = None
    last_horizontal: Optional[Direction] = None

    def update(self, direction: Direction) -> None:
        """Updates the counters based on direction pairs"""
        if direction in (Direction.UP, Direction.DOWN):
            if self.last_vertical is None:
                # First movement of potential pair
                self.last_vertical = direction
            elif (
                self.last_vertical == Direction.UP and direction == Direction.DOWN
            ) or (self.last_vertical == Direction.DOWN and direction == Direction.UP):
                # Complete pair detected
                self.vertical += 1
                self.last_vertical = None
            else:
                # New movement, reset tracking
                self.last_vertical = direction

        elif direction in (Direction.LEFT, Direction.RIGHT):
            if self.last_horizontal is None:
                # First movement of potential pair
                self.last_horizontal = direction
            elif (
                self.last_horizontal == Direction.LEFT and direction == Direction.RIGHT
            ) or (
                self.last_horizontal == Direction.RIGHT and direction == Direction.LEFT
            ):
                # Complete pair detected
                self.horizontal += 1
                self.last_horizontal = None
            else:
                # New movement, reset tracking
                self.last_horizontal = direction

    def to_dict(self) -> Dict[str, int]:
        return {
            "Vertical": self.vertical,  # Up-Down pairs
            "Horizontal": self.horizontal,  # Left-Right pairs
        }

    def reset(self) -> None:
        self.vertical = 0
        self.horizontal = 0
        self.last_vertical = None
        self.last_horizontal = None


# ==============================
# Pose Estimation Core
# ==============================


class HeadPoseEstimator:
    """Handles head pose estimation using MediaPipe Face Mesh"""

    LANDMARK_INDICES = [1, 152, 263, 33, 287, 57, 9]

    MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -400.0, -75.0),
            (-225.0, 170.0, -125.0),
            (225.0, 170.0, -125.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
            (0.0, 350.0, -75.0),
        ],
        dtype=np.float64,
    )

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.prev_pitch = 0.0
        self.prev_yaw = 0.0

    def update_dimensions(self, width: int, height: int):
        self.frame_width = width
        self.frame_height = height
        focal_length = width
        center = (width / 2, height / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )

    def extract_2d_landmarks(self, landmarks) -> np.ndarray:
        points = [
            (landmarks[idx].x * self.frame_width, landmarks[idx].y * self.frame_height)
            for idx in self.LANDMARK_INDICES
        ]
        return np.array(points, dtype=np.float64)

    def smooth_angle(self, prev: float, new: float, alpha: float) -> float:
        return prev * alpha + new * (1 - alpha)

    def estimate_pose(
        self, frame: np.ndarray, smoothing_alpha: float = 0.8
    ) -> Optional[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        image_points = self.extract_2d_landmarks(face_landmarks.landmark)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS, image_points, self.camera_matrix, self.dist_coeffs
        )

        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        proj_mat = np.hstack((rmat, tvec))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)
        pitch, yaw, _ = euler_angles.flatten()

        pitch = self.smooth_angle(self.prev_pitch, pitch, smoothing_alpha)
        yaw = self.smooth_angle(self.prev_yaw, yaw, smoothing_alpha)

        self.prev_pitch = pitch
        self.prev_yaw = yaw

        return pitch, yaw, image_points, rvec, tvec

    def release(self):
        self.face_mesh.close()


# ==============================
# Direction Classifier
# ==============================


class DirectionClassifier:
    def __init__(self, config: ThresholdConfig):
        self.config = config

    def classify(self, pitch: float, yaw: float) -> Direction:
        if (
            abs(pitch) <= self.config.center_pitch
            and abs(yaw) <= self.config.center_yaw
        ):
            return Direction.CENTER

        if pitch < self.config.pitch_up and abs(yaw) <= self.config.center_yaw:
            return Direction.UP

        if pitch > self.config.pitch_down and abs(yaw) <= self.config.center_yaw:
            return Direction.DOWN

        if yaw < self.config.yaw_left and abs(pitch) <= self.config.center_pitch:
            return Direction.RIGHT

        if yaw > self.config.yaw_right and abs(pitch) <= self.config.center_pitch:
            return Direction.LEFT

        return Direction.NOT_DETECTED

    def update_config(self, config: ThresholdConfig):
        self.config = config


# ==============================
# Pattern Recognition
# ==============================


class PatternRecognizer:
    def __init__(self, config: ThresholdConfig):
        self.config = config
        self.sequence: List[Direction] = []
        self.hold_direction: Optional[Direction] = None
        self.hold_count: int = 0
        self.last_rep_time: float = 0.0

    def update(self, current_direction: Direction) -> Optional[Direction]:
        directional = {Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT}

        if current_direction in directional:
            if self.hold_direction != current_direction:
                self.hold_direction = current_direction
                self.hold_count = 1
            else:
                self.hold_count += 1
        else:
            if (
                self.hold_direction is not None
                and self.hold_count >= self.config.hold_required
            ):
                if not self.sequence or self.sequence[-1] != self.hold_direction:
                    self.sequence.append(self.hold_direction)

            self.hold_direction = None
            self.hold_count = 0

            if not self.sequence or self.sequence[-1] != current_direction:
                self.sequence.append(current_direction)

        completed_direction = self._check_pattern()

        if len(self.sequence) > 10:
            self.sequence.pop(0)

        return completed_direction

    def _check_pattern(self) -> Optional[Direction]:
        if len(self.sequence) < 5:
            return None

        # Pattern: CENTER -> NOT_DETECTED -> DIRECTION -> NOT_DETECTED -> CENTER
        if (
            len(self.sequence) >= 5
            and self.sequence[-5] == Direction.CENTER
            and self.sequence[-4] == Direction.NOT_DETECTED
            and self.sequence[-3]
            in {Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT}
            and self.sequence[-2] == Direction.NOT_DETECTED
            and self.sequence[-1] == Direction.CENTER
        ):
            now = time.time()
            if (now - self.last_rep_time) > self.config.min_time_between_reps:
                completed_direction = self.sequence[-3]
                self.last_rep_time = now
                self.sequence = self.sequence[:-5]  # Remove the completed pattern
                return completed_direction

        # Remove oldest entry if no pattern found
        if len(self.sequence) > 10:
            self.sequence.pop(0)

        return None

    def get_hold_count(self) -> int:
        return self.hold_count

    def reset(self):
        self.sequence.clear()
        self.hold_direction = None
        self.hold_count = 0


# ==============================
# Visualization with Threshold Regions
# ==============================


class Visualizer:
    @staticmethod
    def draw_threshold_diagram(
        frame: np.ndarray,
        pitch: float,
        yaw: float,
        config: ThresholdConfig,
        current_direction: Direction,
    ):
        """Draw a visual diagram showing threshold regions and current position"""
        h, w = frame.shape[:2]

        # Diagram position and size
        diagram_size = 200
        margin = 20
        x_start = w - diagram_size - margin
        y_start = margin

        # Draw background
        cv2.rectangle(
            frame,
            (x_start, y_start),
            (x_start + diagram_size, y_start + diagram_size),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            frame,
            (x_start, y_start),
            (x_start + diagram_size, y_start + diagram_size),
            (100, 100, 100),
            2,
        )

        # Center position
        center_x = x_start + diagram_size // 2
        center_y = y_start + diagram_size // 2

        # Scale factors (pixels per degree)
        scale_pitch = diagram_size / 120  # -60 to 60 degrees
        scale_yaw = diagram_size / 120

        # Draw threshold regions with colors
        regions = [
            # UP region
            (
                center_x - int(config.center_yaw * scale_yaw),
                y_start,
                int(config.center_yaw * 2 * scale_yaw),
                int((60 + config.pitch_up) * scale_pitch),
                (50, 50, 200),  # Darker blue
                "UP",
            ),
            # DOWN region
            (
                center_x - int(config.center_yaw * scale_yaw),
                y_start + diagram_size - int((60 - config.pitch_down) * scale_pitch),
                int(config.center_yaw * 2 * scale_yaw),
                int((60 - config.pitch_down) * scale_pitch),
                (200, 50, 50),
                "DOWN",
            ),
            # LEFT region
            (
                x_start,
                center_y - int(config.center_pitch * scale_pitch),
                int((60 + config.yaw_left) * scale_yaw),
                int(config.center_pitch * 2 * scale_pitch),
                (200, 200, 50),  # Darker yellow
                "RIGHT",
            ),
            # RIGHT region
            (
                x_start + diagram_size - int((60 - config.yaw_right) * scale_yaw),
                center_y - int(config.center_pitch * scale_pitch),
                int((60 - config.yaw_right) * scale_yaw),
                int(config.center_pitch * 2 * scale_pitch),
                (50, 200, 50),  # Darker green
                "LEFT",
            ),
            # CENTER region
            (
                center_x - int(config.center_yaw * scale_yaw),
                center_y - int(config.center_pitch * scale_pitch),
                int(config.center_yaw * 2 * scale_yaw),
                int(config.center_pitch * 2 * scale_pitch),
                (150, 150, 150),  # Grey
                "CENTER",
            ),
        ]

        # Draw white background for better visibility
        cv2.rectangle(
            frame,
            (x_start, y_start),
            (x_start + diagram_size, y_start + diagram_size),
            (255, 255, 255),
            -1,
        )

        # Draw regions with transparency
        overlay = frame.copy()
        for x, y, w_rect, h_rect, color, label in regions:
            # Highlight current region
            if current_direction.value == label:
                cv2.rectangle(overlay, (x, y), (x + w_rect, y + h_rect), color, -1)
            else:
                cv2.rectangle(
                    overlay,
                    (x, y),
                    (x + w_rect, y + h_rect),
                    color,
                    2,
                )

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw labels
        label_positions = [
            (center_x - 15, y_start + 20, "UP"),
            (center_x - 25, y_start + diagram_size - 20, "DOWN"),
            (x_start + 10, center_y - 5, "LEFT"),
            (x_start + diagram_size - 50, center_y - 5, "RIGHT"),
            (center_x - 30, center_y - 5, "CENTER"),
        ]

        for lx, ly, text in label_positions:
            # Draw black outline for better visibility
            cv2.putText(
                frame, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )
            # Draw white text
            cv2.putText(
                frame, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Draw current position indicator - invert yaw for mirrored view
        current_x = center_x - int(yaw * scale_yaw)  # Inverted yaw
        current_y = center_y + int(pitch * scale_pitch)

        # Clamp to diagram bounds
        current_x = max(x_start, min(x_start + diagram_size, current_x))
        current_y = max(y_start, min(y_start + diagram_size, current_y))

        cv2.circle(frame, (current_x, current_y), 8, (0, 255, 255), -1)
        cv2.circle(frame, (current_x, current_y), 10, (255, 255, 255), 2)

    @staticmethod
    def draw_angles_and_direction(
        frame: np.ndarray,
        pitch: float,
        yaw: float,
        direction: Direction,
        hold_count: int,
    ):
        """Draw angle values and current direction"""
        cv2.putText(
            frame,
            f"Pitch: {pitch:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Yaw: {yaw:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        color = (0, 255, 0) if direction != Direction.NOT_DETECTED else (0, 0, 255)
        cv2.putText(
            frame,
            f"Direction: {direction.value}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Hold: {hold_count}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )

    @staticmethod
    def draw_landmarks(frame: np.ndarray, image_points: np.ndarray, indices: List[int]):
        for (x, y), idx in zip(image_points, indices):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
            cv2.putText(
                frame,
                f"{idx}",
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

    @staticmethod
    def draw_model_points(
        frame: np.ndarray,
        model_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ):
        projected, _ = cv2.projectPoints(
            model_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = np.squeeze(projected)

        for i, (x, y) in enumerate(projected):
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)
            cv2.putText(
                frame,
                f"M{i}",
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )


# ==============================
# Main Application
# ==============================


class HeadPoseTracker:
    def __init__(self):
        self.config = ThresholdConfig()
        self.estimator = HeadPoseEstimator()
        self.classifier = DirectionClassifier(self.config)
        self.recognizer = PatternRecognizer(self.config)
        self.rep_counter = RepCounter()
        self.visualizer = Visualizer()
        self.developer_mode = False
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_direction = Direction.NOT_DETECTED

    def update_thresholds(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.classifier.update_config(self.config)
        self.recognizer.config = self.config

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame

        h, w = frame.shape[:2]
        if w != self.estimator.frame_width or h != self.estimator.frame_height:
            self.estimator.update_dimensions(w, h)

        pose_data = self.estimator.estimate_pose(frame, self.config.smoothing_alpha)

        if pose_data:
            pitch, yaw, image_points, rvec, tvec = pose_data
            self.current_pitch = pitch
            self.current_yaw = yaw

            current_direction = self.classifier.classify(pitch, yaw)
            self.current_direction = current_direction

            completed_direction = self.recognizer.update(current_direction)
            if completed_direction:
                self.rep_counter.update(completed_direction)

            if self.developer_mode:
                # Draw angles, direction, and hold count
                self.visualizer.draw_angles_and_direction(
                    frame,
                    pitch,
                    yaw,
                    current_direction,
                    self.recognizer.get_hold_count(),
                )

                # Draw threshold diagram with highlighted regions
                self.visualizer.draw_threshold_diagram(
                    frame, pitch, yaw, self.config, current_direction
                )

                # Draw landmarks and model points
                self.visualizer.draw_landmarks(
                    frame, image_points, HeadPoseEstimator.LANDMARK_INDICES
                )
                self.visualizer.draw_model_points(
                    frame,
                    HeadPoseEstimator.MODEL_POINTS,
                    rvec,
                    tvec,
                    self.estimator.camera_matrix,
                    self.estimator.dist_coeffs,
                )

        return frame

    def get_rep_counts(self) -> Dict[str, int]:
        return self.rep_counter.to_dict()

    def reset_counters(self):
        self.rep_counter.reset()
        self.recognizer.reset()

    def release(self):
        self.estimator.release()


# ==============================
# Kivy GUI Application
# ==============================


class HeadPoseApp(App):
    def build(self):
        # Set background color for the app
        Window.clearcolor = (0.1, 0.1, 0.1, 1)  # Dark background

        self.tracker = HeadPoseTracker()
        self.capture = cv2.VideoCapture(0)

        # Main layout with background
        main_layout = BoxLayout(orientation="horizontal", padding=15, spacing=15)
        with main_layout.canvas.before:
            Color(0.15, 0.15, 0.15, 1)  # Slightly lighter than background
            self.rect = Rectangle(size=main_layout.size, pos=main_layout.pos)
        main_layout.bind(size=self._update_rect, pos=self._update_rect)

        # Left side - Video and controls
        left_layout = BoxLayout(orientation="vertical", size_hint=(0.7, 1), spacing=15)

        # Video display with border
        video_container = BoxLayout(size_hint=(1, 0.8))
        with video_container.canvas.before:
            Color(0.2, 0.2, 0.2, 1)  # Border color
            self.video_rect = Rectangle(
                size=video_container.size, pos=video_container.pos
            )
        video_container.bind(size=self._update_video_rect, pos=self._update_video_rect)

        self.img = Image(size_hint=(1, 1))
        video_container.add_widget(self.img)
        left_layout.add_widget(video_container)

        # Rep counters with enhanced styling
        counters_layout = BoxLayout(size_hint=(1, 0.12), spacing=15, padding=15)
        with counters_layout.canvas.before:
            Color(0.2, 0.2, 0.2, 1)  # Background color for counter section
            self.counter_rect = Rectangle(
                size=counters_layout.size, pos=counters_layout.pos
            )
        counters_layout.bind(
            size=self._update_counter_rect, pos=self._update_counter_rect
        )

        counter_style = {
            "font_size": "24sp",
            "bold": True,
            "color": (1, 1, 1, 1),  # White text
            "outline_width": 2,
            "outline_color": (0, 0, 0, 1),  # Black outline
        }

        self.vertical_label = Label(text="Up-Down: 0", **counter_style)
        self.horizontal_label = Label(text="Left-Right: 0", **counter_style)

        for label in [
            self.vertical_label,
            self.horizontal_label,
        ]:
            container = BoxLayout(padding=5)
            with container.canvas.before:
                Color(
                    0.25, 0.25, 0.25, 1
                )  # Slightly lighter background for each counter
                RoundedRectangle(size=container.size, pos=container.pos, radius=[10])
            container.bind(
                size=lambda inst, val, l=label: self._update_label_rect(inst, val, l)
            )
            container.add_widget(label)
            counters_layout.add_widget(container)

        left_layout.add_widget(counters_layout)

        # Control buttons with enhanced styling
        buttons_layout = BoxLayout(size_hint=(1, 0.1), spacing=15, padding=10)

        button_style = {
            "font_size": "18sp",
            "background_normal": "",
            "background_color": (0.2, 0.6, 0.8, 1),  # Blue color
            "color": (1, 1, 1, 1),  # White text
            "bold": True,
            "size_hint_y": None,
            "height": "48dp",
        }

        dev_btn = Button(text="Toggle Dev Mode", **button_style)
        dev_btn.bind(on_press=self.toggle_dev_mode)

        reset_btn = Button(text="Reset Counters", **button_style)
        reset_btn.background_color = (0.8, 0.2, 0.2, 1)  # Red color
        reset_btn.bind(on_press=self.reset_counters)

        fullscreen_btn = Button(text="Toggle Fullscreen", **button_style)
        fullscreen_btn.background_color = (0.2, 0.8, 0.2, 1)  # Green color
        fullscreen_btn.bind(on_press=self.toggle_fullscreen)

        buttons_layout.add_widget(dev_btn)
        buttons_layout.add_widget(reset_btn)
        buttons_layout.add_widget(fullscreen_btn)
        left_layout.add_widget(buttons_layout)

        main_layout.add_widget(left_layout)

        # Right side - Threshold controls with enhanced styling
        right_layout = BoxLayout(
            orientation="vertical", size_hint=(0.3, 1), padding=15, spacing=10
        )
        with right_layout.canvas.before:
            Color(0.2, 0.2, 0.2, 1)
            self.right_rect = Rectangle(size=right_layout.size, pos=right_layout.pos)
        right_layout.bind(size=self._update_right_rect, pos=self._update_right_rect)

        # Title with enhanced styling
        title_box = BoxLayout(size_hint=(1, 0.08), padding=[0, 5])
        with title_box.canvas.before:
            Color(0.3, 0.3, 0.3, 1)
            RoundedRectangle(size=title_box.size, pos=title_box.pos, radius=[10])
        title_box.bind(size=self._update_title_rect, pos=self._update_title_rect)

        title = Label(
            text="Threshold Controls",
            font_size="24sp",
            size_hint=(1, 1),
            bold=True,
            color=(1, 1, 1, 1),
        )
        title_box.add_widget(title)
        right_layout.add_widget(title_box)

        # Create sliders for all thresholds with enhanced styling
        self.sliders = {}
        slider_configs = [
            ("pitch_up", "Pitch Up", -60, 0, -35),
            ("pitch_down", "Pitch Down", 0, 60, 35),
            ("center_pitch", "Center Pitch", 0, 30, 15),
            ("yaw_left", "Yaw Left", -60, 0, -45),
            ("yaw_right", "Yaw Right", 0, 60, 45),
            ("center_yaw", "Center Yaw", 0, 30, 10),
            ("hold_required", "Hold Frames", 10, 60, 30),
            ("min_time_between_reps", "Min Time (s)", 0.1, 2.0, 0.8),
            ("smoothing_alpha", "Smoothing", 0.1, 1.0, 0.8),
        ]

        for key, name, min_val, max_val, default in slider_configs:
            slider_box = BoxLayout(
                orientation="vertical", size_hint=(1, 0.1), padding=[5, 2]
            )
            with slider_box.canvas.before:
                Color(0.25, 0.25, 0.25, 1)
                RoundedRectangle(size=slider_box.size, pos=slider_box.pos, radius=[5])
            slider_box.bind(size=self._update_slider_rect, pos=self._update_slider_rect)

            label = Label(
                text=f"{name}: {default}",
                font_size="16sp",
                size_hint=(1, 0.4),
                color=(0.9, 0.9, 0.9, 1),
                bold=True,
            )

            slider = Slider(
                min=min_val,
                max=max_val,
                value=default,
                size_hint=(1, 0.6),
                cursor_size=(20, 20),
                background_width="8dp",
                value_track=True,
                value_track_color=[0.2, 0.6, 0.8, 1],  # Blue track
            )

            slider.bind(
                value=lambda instance, value, l=label, n=name, k=key: self.update_slider(
                    l, n, k, value
                )
            )

            slider_box.add_widget(label)
            slider_box.add_widget(slider)
            right_layout.add_widget(slider_box)
            self.sliders[key] = (slider, label)

        main_layout.add_widget(right_layout)

        # Schedule video update
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        # Start in fullscreen mode
        Window.fullscreen = True
        # Window.size = (1920, 1080)  # Default size when exiting fullscreen
        return main_layout

    def update_slider(self, label, name, key, value):
        label.text = f"{name}: {value:.2f}"
        self.tracker.update_thresholds(**{key: value})

    def toggle_dev_mode(self, instance):
        self.tracker.developer_mode = not self.tracker.developer_mode

    def reset_counters(self, instance):
        self.tracker.reset_counters()

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Flip horizontally for mirror effect before processing
            frame = cv2.flip(frame, 1)  # 1 for horizontal flip

            # Process frame with overlays
            frame = self.tracker.process_frame(frame)

            # Convert to Kivy texture (flip vertically because Kivy coordinate system is inverted)
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
            )
            texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.img.texture = texture

            # Update counters
            counts = self.tracker.get_rep_counts()
            self.vertical_label.text = f"Up-Down: {counts['Vertical']}"
            self.horizontal_label.text = f"Left-Right: {counts['Horizontal']}"

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def _update_video_rect(self, instance, value):
        self.video_rect.pos = instance.pos
        self.video_rect.size = instance.size

    def _update_counter_rect(self, instance, value):
        self.counter_rect.pos = instance.pos
        self.counter_rect.size = instance.size

    def _update_label_rect(self, instance, value, label):
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(0.25, 0.25, 0.25, 1)
            RoundedRectangle(size=instance.size, pos=instance.pos, radius=[10])

    def _update_right_rect(self, instance, value):
        self.right_rect.pos = instance.pos
        self.right_rect.size = instance.size

    def _update_title_rect(self, instance, value):
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(0.3, 0.3, 0.3, 1)
            RoundedRectangle(size=instance.size, pos=instance.pos, radius=[10])

    def _update_slider_rect(self, instance, value):
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(0.25, 0.25, 0.25, 1)
            RoundedRectangle(size=instance.size, pos=instance.pos, radius=[5])

    def toggle_fullscreen(self, instance):
        Window.fullscreen = not Window.fullscreen
        instance.background_color = (
            (0.4, 1, 0.4, 1) if Window.fullscreen else (0.2, 0.8, 0.2, 1)
        )
        instance.text = "Exit Fullscreen" if Window.fullscreen else "Enter Fullscreen"

    def on_stop(self):
        self.capture.release()
        self.tracker.release()


if __name__ == "__main__":
    HeadPoseApp().run()

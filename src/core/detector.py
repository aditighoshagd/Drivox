"""
Driver Drowsiness and Distraction Detection Core Engine
Implements real-time facial landmark analysis for driver fatigue monitoring.
Algorithms:
  - Eye Aspect Ratio (EAR) for micro-sleep detection (Soukupová and Čech, 2016)
  - Mouth Aspect Ratio (MAR) for yawn frequency tracking
  - Head Pose Estimation (Pitch, Yaw, Roll) via Perspective-n-Point (PnP) for distraction
  - PERCLOS (Percentage of Eye Closure) sliding window fatigue index
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
import time
from collections import deque


@dataclass
class DetectionResult:
    """Structured data container for per-frame detection telemetry."""
    face_detected: bool = False
    ear: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    head_pose: Dict[str, float] = field(default_factory=lambda: {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
    is_drowsy: bool = False
    is_yawning: bool = False
    is_distracted: bool = False
    fatigue_score: float = 0.0  # 0.0 (fully alert) to 1.0 (severely fatigued)
    status: str = "ALERT"        # ALERT, YAWN_WARNING, DROWSY, DISTRACTED
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class FatigueDetector:
    """
    Core Computer Vision engine for calculating driver fatigue and attention metrics.
    Employs MediaPipe FaceMesh (468 landmarks) with real-time geometric calculations.
    """

    # MediaPipe 468 landmark indices for eyes and mouth
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    MOUTH_INDICES = [61, 291, 13, 14, 78, 308] # Outer mouth / lip landmarks: corners and upper/lower centers

    # Key landmark indices for 3D Head Pose estimation (PnP)
    # Nose tip, Chin, Left eye left corner, Right eye right corner, Left mouth corner, Right mouth corner
    HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]

    # Standard 3D facial model coordinates (in millimeters)
    MODEL_3D_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    def __init__(
        self,
        ear_threshold: float = 0.25,
        ear_consec_frames: int = 20,
        mar_threshold: float = 0.65,
        mar_consec_frames: int = 15,
        perclos_window_size: int = 60,
        head_pitch_threshold: float = 20.0,
        head_yaw_threshold: float = 28.0
    ):
        self.ear_threshold = ear_threshold
        self.ear_consec_frames = ear_consec_frames
        self.mar_threshold = mar_threshold
        self.mar_consec_frames = mar_consec_frames
        self.perclos_window_size = perclos_window_size
        self.head_pitch_threshold = head_pitch_threshold
        self.head_yaw_threshold = head_yaw_threshold

        # Sliding window state
        self.closed_eyes_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.ear_history: deque = deque(maxlen=perclos_window_size)
        self.eye_closed_history: deque = deque(maxlen=perclos_window_size)

        # Lazy loaded mediapipe solution
        self._mp_face_mesh = None
        self._face_mesh = None

    def _init_mediapipe(self):
        """Initializes MediaPipe Face Mesh on first call."""
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._mp_face_mesh = mp.solutions.face_mesh
                self._face_mesh = self._mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except ImportError:
                self._face_mesh = False

    @staticmethod
    def euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
        """Calculates Euclidean distance between two 2D/3D points."""
        return float(np.linalg.norm(np.asarray(pt1) - np.asarray(pt2)))

    def calculate_ear(self, landmarks: List[Tuple[int, int]], eye_indices: List[int]) -> float:
        """
        Calculates Eye Aspect Ratio (EAR) based on 6 landmark points.
        Formula:
            EAR = (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)
        """
        p1 = np.array(landmarks[eye_indices[0]])
        p2 = np.array(landmarks[eye_indices[1]])
        p3 = np.array(landmarks[eye_indices[2]])
        p4 = np.array(landmarks[eye_indices[3]])
        p5 = np.array(landmarks[eye_indices[4]])
        p6 = np.array(landmarks[eye_indices[5]])

        # Vertical distances
        v1 = self.euclidean_distance(p2, p6)
        v2 = self.euclidean_distance(p3, p5)
        # Horizontal distance
        h = self.euclidean_distance(p1, p4)

        if h == 0.0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def calculate_mar(self, landmarks: List[Tuple[int, int]], mouth_indices: List[int]) -> float:
        """
        Calculates Mouth Aspect Ratio (MAR) for yawn detection.
        Formula:
            MAR = (|p3 - p4|) / (|p1 - p2|)
        """
        p1 = np.array(landmarks[mouth_indices[0]])  # left corner
        p2 = np.array(landmarks[mouth_indices[1]])  # right corner
        p3 = np.array(landmarks[mouth_indices[2]])  # upper lip center
        p4 = np.array(landmarks[mouth_indices[3]])  # lower lip center

        v = self.euclidean_distance(p3, p4)
        h = self.euclidean_distance(p1, p2)

        if h == 0.0:
            return 0.0
        return v / h

    def estimate_head_pose(
        self,
        landmarks: List[Tuple[int, int]],
        img_w: int,
        img_h: int
    ) -> Dict[str, float]:
        """
        Estimates Pitch, Yaw, and Roll using Perspective-n-Point (cv2.solvePnP).
        """
        if cv2 is None:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        image_points = np.array([
            landmarks[idx] for idx in self.HEAD_POSE_INDICES
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_3D_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

        # angles: pitch, yaw, roll in degrees
        return {
            "pitch": round(float(angles[0] * 360), 2),
            "yaw": round(float(angles[1] * 360), 2),
            "roll": round(float(angles[2] * 360), 2)
        }

    def process_frame(self, frame: np.ndarray) -> Tuple[DetectionResult, Optional[List[Tuple[int, int]]]]:
        """
        Processes a single BGR video frame and returns the fatigue detection result.
        """
        start_time = time.perf_counter()
        result = DetectionResult(face_detected=False)

        if cv2 is None:
            result.latency_ms = (time.perf_counter() - start_time) * 1000.0
            return result, None

        self._init_mediapipe()
        if not self._face_mesh:
            result.latency_ms = (time.perf_counter() - start_time) * 1000.0
            return result, None

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = self._face_mesh.process(rgb_frame)

        if not mesh_results.multi_face_landmarks:
            result.latency_ms = (time.perf_counter() - start_time) * 1000.0
            return result, None

        face_landmarks = mesh_results.multi_face_landmarks[0]
        landmarks_2d = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

        result.face_detected = True

        # Calculate EAR
        left_ear = self.calculate_ear(landmarks_2d, self.LEFT_EYE_INDICES)
        right_ear = self.calculate_ear(landmarks_2d, self.RIGHT_EYE_INDICES)
        ear = (left_ear + right_ear) / 2.0
        result.left_ear = round(left_ear, 4)
        result.right_ear = round(right_ear, 4)
        result.ear = round(ear, 4)

        # Calculate MAR
        mar = self.calculate_mar(landmarks_2d, self.MOUTH_INDICES)
        result.mar = round(mar, 4)

        # Head Pose
        head_pose = self.estimate_head_pose(landmarks_2d, w, h)
        result.head_pose = head_pose

        # Sliding window history & PERCLOS
        is_eye_closed = ear < self.ear_threshold
        self.eye_closed_history.append(1 if is_eye_closed else 0)
        self.ear_history.append(ear)

        perclos = (sum(self.eye_closed_history) / len(self.eye_closed_history)) if self.eye_closed_history else 0.0
        result.perclos = round(perclos, 4)

        # Consecutive frame triggers
        if is_eye_closed:
            self.closed_eyes_counter += 1
        else:
            self.closed_eyes_counter = max(0, self.closed_eyes_counter - 1)

        if mar > self.mar_threshold:
            self.yawn_counter += 1
        else:
            self.yawn_counter = max(0, self.yawn_counter - 1)

        # Head pose distraction checks
        is_distracted = (
            abs(head_pose["yaw"]) > self.head_yaw_threshold or
            abs(head_pose["pitch"]) > self.head_pitch_threshold
        )
        if is_distracted:
            self.distraction_counter += 1
        else:
            self.distraction_counter = max(0, self.distraction_counter - 1)

        # State evaluations
        result.is_drowsy = (self.closed_eyes_counter >= self.ear_consec_frames) or (perclos > 0.40)
        result.is_yawning = self.yawn_counter >= self.mar_consec_frames
        result.is_distracted = self.distraction_counter >= 15

        # Fatigue score: composite indicator (0.0 to 1.0)
        ear_factor = max(0.0, min(1.0, (self.ear_threshold - ear) / (self.ear_threshold * 0.7))) if is_eye_closed else 0.0
        perclos_factor = min(1.0, perclos / 0.5)
        yawn_factor = min(1.0, self.yawn_counter / self.mar_consec_frames)

        fatigue_score = (0.5 * perclos_factor) + (0.3 * ear_factor) + (0.2 * yawn_factor)
        result.fatigue_score = round(min(1.0, fatigue_score), 3)

        # Determine overall categorical status
        if result.is_drowsy:
            result.status = "DROWSY"
        elif result.is_distracted:
            result.status = "DISTRACTED"
        elif result.is_yawning:
            result.status = "YAWN_WARNING"
        elif result.fatigue_score > 0.4:
            result.status = "FATIGUE_ADVISORY"
        else:
            result.status = "ALERT"

        result.latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return result, landmarks_2d

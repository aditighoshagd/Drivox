"""
Data Preprocessing and Transformation Feature Pipeline
Handles real-time image transformations, contrast enhancements for nighttime driving,
and tensor preparation for deep learning inference.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None


class ImagePreprocessingPipeline:
    """
    Standardized preprocessing pipeline ensuring high data quality and illumination robustness.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 480), gamma: float = 1.3):
        self.target_size = target_size
        self.gamma = gamma
        if cv2 is not None:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self.clahe = None
        self._init_gamma_lut(gamma)

    def _init_gamma_lut(self, gamma: float):
        """Precomputes lookup table for fast non-linear gamma luminance correction."""
        inv_gamma = 1.0 / max(0.1, gamma)
        self.gamma_lut = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
        ]).astype(np.uint8)

    def validate_frame(self, frame: Any) -> bool:
        """Validates that input frame meets dimensionality, channel, and non-empty criteria."""
        if frame is None or not isinstance(frame, np.ndarray):
            return False
        if frame.size == 0 or len(frame.shape) != 3:
            return False
        if frame.shape[2] != 3:
            return False
        return True

    def adjust_gamma(self, image: np.ndarray) -> np.ndarray:
        """Applies fast non-linear gamma lookup table to compensate for night driving."""
        if cv2 is not None:
            return cv2.LUT(image, self.gamma_lut)
        return self.gamma_lut[image]

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Applies CLAHE on luminance channel in YCrCb color space for uneven vehicle lighting."""
        if cv2 is not None and self.clahe is not None:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return image

    def preprocess_stream_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Executes end-to-end preprocessing pipeline for live video frames:
        1. Dimension validation
        2. Bilinear resize for consistent inference latency
        3. Gamma luminance compensation
        """
        if not self.validate_frame(frame):
            raise ValueError("Invalid frame passed to preprocessing pipeline.")

        # Resize to fixed operational resolution
        if cv2 is not None:
            processed = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        else:
            h, w = frame.shape[:2]
            target_w, target_h = self.target_size
            y_indices = np.linspace(0, h - 1, target_h).astype(int)
            x_indices = np.linspace(0, w - 1, target_w).astype(int)
            processed = frame[y_indices[:, None], x_indices]

        # Apply gamma
        processed = self.adjust_gamma(processed)
        return processed

    def extract_eye_roi(
        self,
        frame: np.ndarray,
        eye_landmarks: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Optional[np.ndarray]:
        """
        Extracts, pads, and normalizes eye bounding box for deep learning classifier.
        Output is normalized to [0, 1] range float32.
        """
        if len(eye_landmarks) == 0:
            return None

        x_min, y_min = np.min(eye_landmarks, axis=0)
        x_max, y_max = np.max(eye_landmarks, axis=0)

        # Apply 25% margin
        w_margin = int((x_max - x_min) * 0.25)
        h_margin = int((y_max - y_min) * 0.25)

        h, w, _ = frame.shape
        x1 = max(0, int(x_min - w_margin))
        y1 = max(0, int(y_min - h_margin))
        x2 = min(w, int(x_max + w_margin))
        y2 = min(h, int(y_max + h_margin))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None

        if cv2 is not None:
            resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        else:
            h_c, w_c = crop.shape[:2]
            target_w, target_h = target_size
            y_indices = np.linspace(0, h_c - 1, target_h).astype(int)
            x_indices = np.linspace(0, w_c - 1, target_w).astype(int)
            resized = crop[y_indices[:, None], x_indices]

        normalized = resized.astype(np.float32) / 255.0
        return normalized

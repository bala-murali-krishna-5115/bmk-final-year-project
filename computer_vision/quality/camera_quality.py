import cv2
import numpy as np
from typing import Dict, Any, Optional

class CameraQualityEngine:
    """
    Evaluates image frame quality for brightness, blur/sharpness, face size, and motion stability.
    Returns structured heuristic quality scores.
    """

    def __init__(self, min_brightness: float = 40.0, max_brightness: float = 220.0, min_blur_var: float = 80.0):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_blur_var = min_blur_var
        self.prev_gray: Optional[np.ndarray] = None

    def evaluate_frame(self, frame: np.ndarray, face_bbox: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Calculates frame quality heuristics.

        Args:
            frame (np.ndarray): BGR image frame.
            face_bbox (Optional[Dict[str, int]]): Face bounding box with keys 'x', 'y', 'width', 'height'.

        Returns:
            Dict[str, Any]: Quality evaluation dictionary containing scores and status.
        """
        if frame is None or frame.size == 0:
            return {
                "brightness_score": 0.0,
                "blur_score": 0.0,
                "face_size_score": 0.0,
                "stability_score": 0.0,
                "overall_quality": 0.0,
                "status": "invalid_frame"
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Brightness score (Heuristic: ideal mean brightness between 80 and 180)
        mean_brightness = float(np.mean(gray))
        if self.min_brightness <= mean_brightness <= self.max_brightness:
            # Scale 0 to 100 based on distance from range edges
            if mean_brightness < 120:
                brightness_score = float(np.interp(mean_brightness, [self.min_brightness, 120], [50, 100]))
            else:
                brightness_score = float(np.interp(mean_brightness, [120, self.max_brightness], [100, 50]))
        else:
            brightness_score = 20.0

        # 2. Blur / Sharpness score (Laplacian variance: >150 is sharp, <50 is blurry)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = float(np.clip(np.interp(blur_var, [20.0, 200.0], [0.0, 100.0]), 0.0, 100.0))

        # 3. Face size score (Heuristic: face area should be 15% to 60% of frame area)
        face_size_score = 0.0
        if face_bbox is not None:
            face_area = face_bbox.get('width', 0) * face_bbox.get('height', 0)
            frame_area = w * h
            size_ratio = face_area / max(frame_area, 1)
            face_size_score = float(np.clip(np.interp(size_ratio, [0.05, 0.25, 0.60, 0.85], [10, 100, 100, 30]), 0, 100))

        # 4. Motion Stability score (Frame difference vs previous frame)
        stability_score = 100.0
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            frame_diff = float(np.mean(cv2.absdiff(gray, self.prev_gray)))
            stability_score = float(np.clip(np.interp(frame_diff, [0.0, 25.0], [100.0, 10.0]), 0.0, 100.0))
        self.prev_gray = gray.copy()

        # Composite heuristic quality score
        overall_quality = float(np.round(
            0.30 * brightness_score +
            0.35 * blur_score +
            0.20 * face_size_score +
            0.15 * stability_score,
            2
        ))

        status = "good" if overall_quality >= 65.0 else "low_quality"

        return {
            "brightness_mean": round(mean_brightness, 2),
            "brightness_score": round(brightness_score, 2),
            "blur_variance": round(blur_var, 2),
            "blur_score": round(blur_score, 2),
            "face_size_score": round(face_size_score, 2),
            "stability_score": round(stability_score, 2),
            "overall_quality": overall_quality,
            "status": status
        }

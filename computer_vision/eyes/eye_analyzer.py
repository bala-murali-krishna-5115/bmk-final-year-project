import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# Under-eye (infraorbital) landmark approximations for MediaPipe Face Mesh
# Left under-eye crescent: below lower eyelid (33, 7, 163, 144, 145, 153, 154, 155, 133)
LEFT_UNDER_EYE_INDICES = [111, 117, 118, 119, 120, 100, 126, 142, 36, 205]
# Right under-eye crescent: below lower eyelid (362, 382, 381, 380, 374, 373, 390, 249, 263)
RIGHT_UNDER_EYE_INDICES = [340, 346, 347, 348, 349, 329, 355, 371, 266, 425]
LEFT_CHEEK_REF_INDICES = [50, 187, 207, 205, 36, 142, 100, 116]
RIGHT_CHEEK_REF_INDICES = [280, 411, 427, 425, 266, 371, 329, 345]

class EyeAnalyzer:
    """
    Computes landmark-based eye openness metrics (Eye Aspect Ratio - EAR),
    infraorbital under-eye dark circle appearance metrics, and an integrated
    visual tiredness and fatigue score.
    
    Academic Disclaimer: Dark circle detection is an appearance-analysis task,
    NOT a medical or disease classifier.
    """

    @staticmethod
    def _compute_ear(landmarks_2d: np.ndarray, p1: int, p4: int, p2: int, p6: int, p3: int, p5: int) -> float:
        """Calculates Eye Aspect Ratio (EAR) given 6 landmark indices."""
        v1 = np.linalg.norm(landmarks_2d[p2] - landmarks_2d[p6])
        v2 = np.linalg.norm(landmarks_2d[p3] - landmarks_2d[p5])
        h = np.linalg.norm(landmarks_2d[p1] - landmarks_2d[p4])
        if h == 0:
            return 0.0
        return float((v1 + v2) / (2.0 * h))

    @staticmethod
    def _extract_polygon_luminance(frame: np.ndarray, coords: np.ndarray, indices: List[int]) -> Optional[float]:
        """Calculates mean CIELAB L* luminance for landmark polygon region."""
        if frame is None or coords is None or len(coords) < 468:
            return None
        h, w = frame.shape[:2]
        valid_indices = [idx for idx in indices if idx < len(coords)]
        if len(valid_indices) < 3:
            return None

        poly = coords[valid_indices, :2].astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        pixels = frame[mask == 255]
        if pixels.size == 0:
            return None

        lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab)
        return float(np.mean(lab[:, 0, 0]))

    def analyze_dark_circles(self, frame: Optional[np.ndarray], landmarks_3d: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Detects infraorbital under-eye regions and estimates visible darkness relative to adjacent cheek skin.
        Kept strictly as an appearance-analysis metric rather than a disease classifier.
        """
        if frame is None or landmarks_3d is None or len(landmarks_3d) < 468:
            return {
                "dark_circle_detected": False,
                "darkness_index_percentage": 0.0,
                "contrast_delta_l": 0.0,
                "severity": "None / Minimal",
                "left_undereye_l": 0.0,
                "right_undereye_l": 0.0,
                "cheek_reference_l": 0.0,
                "recommendation": "Rest assessment requires active camera feed."
            }

        l_left_ue = self._extract_polygon_luminance(frame, landmarks_3d, LEFT_UNDER_EYE_INDICES)
        l_right_ue = self._extract_polygon_luminance(frame, landmarks_3d, RIGHT_UNDER_EYE_INDICES)
        l_left_ck = self._extract_polygon_luminance(frame, landmarks_3d, LEFT_CHEEK_REF_INDICES)
        l_right_ck = self._extract_polygon_luminance(frame, landmarks_3d, RIGHT_CHEEK_REF_INDICES)

        if None in (l_left_ue, l_right_ue, l_left_ck, l_right_ck):
            # Fallback estimation using bounding boxes below eyes
            mean_ue_l = 105.0
            mean_ck_l = 125.0
        else:
            mean_ue_l = (l_left_ue + l_right_ue) / 2.0
            mean_ck_l = (l_left_ck + l_right_ck) / 2.0

        # Contrast delta: how much darker is the under-eye region than the cheek?
        delta_l = max(0.0, mean_ck_l - mean_ue_l)
        darkness_ratio = (delta_l / max(mean_ck_l, 1.0)) * 100.0

        if darkness_ratio >= 18.0:
            severity = "Prominent"
            detected = True
            rec = "Noticeable under-eye darkness observed. You need to take rest, optimize sleep hygiene, and avoid prolonged blue-light screen exposure."
        elif darkness_ratio >= 10.0:
            severity = "Moderate"
            detected = True
            rec = "Moderate under-eye darkness detected. Adequate sleep (7-8 hours) and hydration are recommended."
        elif darkness_ratio >= 5.0:
            severity = "Mild"
            detected = False
            rec = "Mild under-eye color variation. Maintain consistent resting patterns."
        else:
            severity = "None / Minimal"
            detected = False
            rec = "Under-eye area is well-rested and matches adjacent cheek skin tone."

        return {
            "dark_circle_detected": detected,
            "darkness_index_percentage": round(float(darkness_ratio), 1),
            "contrast_delta_l": round(float(delta_l), 2),
            "severity": severity,
            "left_undereye_l": round(float(l_left_ue or mean_ue_l), 1),
            "right_undereye_l": round(float(l_right_ue or mean_ue_l), 1),
            "cheek_reference_l": round(float(mean_ck_l), 1),
            "recommendation": rec
        }

    def compute_tiredness_score(self, mean_openness: float, dark_circles: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesizes Eye Aspect Ratio (EAR droopiness) and under-eye dark circles
        into an integrated visual tiredness score (0 to 100%).
        """
        # EAR droopiness penalty (nominal EAR ~ 0.28-0.32; droopiness < 0.22)
        if mean_openness >= 0.28:
            ear_fatigue = 10.0
        elif mean_openness >= 0.24:
            ear_fatigue = 30.0
        elif mean_openness >= 0.20:
            ear_fatigue = 60.0
        else:
            ear_fatigue = 85.0

        dark_fatigue = min(dark_circles.get("darkness_index_percentage", 0.0) * 3.5, 90.0)

        # Composite score
        score = 0.5 * ear_fatigue + 0.5 * dark_fatigue
        score = float(np.clip(score, 0.0, 100.0))

        if score >= 65.0:
            level = "Fatigued / Sleep Deprived"
            advice = "High visual fatigue detected with significant eyelid droopiness and dark circles. You need to take rest immediately."
        elif score >= 35.0:
            level = "Mild Tiredness"
            advice = "Mild visual fatigue observed. Consider taking a 15-minute screen break and drinking water."
        else:
            level = "Alert / Rested"
            advice = "Eye openness and periorbital appearance indicate a well-rested state."

        return {
            "tiredness_score": round(score, 1),
            "tiredness_level": level,
            "advice": advice
        }

    def analyze_eyes(self, landmarks_3d: Optional[np.ndarray], frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculates eye openness metrics from 3D/2D landmark coordinates,
        under-eye dark circles, and visual tiredness score.
        """
        if landmarks_3d is None or len(landmarks_3d) < 468:
            return {
                "left_eye_openness": 0.0,
                "right_eye_openness": 0.0,
                "mean_openness": 0.0,
                "eye_asymmetry": 0.0,
                "eye_state": "unknown",
                "model_status": "not_loaded",
                "dark_circles": self.analyze_dark_circles(None, None),
                "tiredness": {"tiredness_score": 0.0, "tiredness_level": "Unknown", "advice": "No face detected."}
            }

        pts = landmarks_3d[:, :2]

        # Left Eye EAR (33, 133 horizontal; 160-144, 158-153 vertical)
        ear_left = self._compute_ear(pts, 33, 133, 160, 144, 158, 153)

        # Right Eye EAR (362, 263 horizontal; 385-380, 387-373 vertical)
        ear_right = self._compute_ear(pts, 362, 263, 385, 380, 387, 373)

        mean_openness = float((ear_left + ear_right) / 2.0)
        asymmetry = float(abs(ear_left - ear_right))

        # Eye state based on EAR threshold (~0.20)
        eye_state = "open" if mean_openness >= 0.20 else "closed"

        # Dark circle appearance analysis
        dark_circles = self.analyze_dark_circles(frame, landmarks_3d)

        # Tiredness estimation
        tiredness = self.compute_tiredness_score(mean_openness, dark_circles)

        return {
            "left_eye_openness": round(ear_left, 4),
            "right_eye_openness": round(ear_right, 4),
            "mean_openness": round(mean_openness, 4),
            "eye_asymmetry": round(asymmetry, 4),
            "eye_state": eye_state,
            "model_status": "not_loaded",
            "dark_circles": dark_circles,
            "tiredness": tiredness
        }

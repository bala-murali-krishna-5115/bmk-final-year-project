import cv2
import numpy as np
from typing import Dict, Any, Optional, List

class HairAnalyzer:
    """
    Analyzes visible hair, hairline transition, scalp coverage density,
    and texture/graying characteristics from the superior facial region.
    
    Academic Disclaimer: Appearance analysis only. Not a medical trichology diagnosis.
    """

    def analyze_hair(self, frame: Optional[np.ndarray], landmarks_3d: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Extracts scalp/hair ROI immediately superior to the forehead landmarks (landmark 10)
        and computes density, hairline stability, and texture metrics.
        """
        if frame is None or landmarks_3d is None or len(landmarks_3d) < 468:
            return {
                "hair_detected": False,
                "hair_density_score": 0.0,
                "hairline_stability": "Normal",
                "hair_texture_coarseness": 0.0,
                "graying_ratio_percentage": 0.0,
                "summary": "No facial landmarks detected for hair analysis."
            }

        h, w = frame.shape[:2]
        # Topmost forehead landmark (landmark 10)
        top_forehead = landmarks_3d[10, :2]
        fx, fy = int(top_forehead[0]), int(top_forehead[1])

        # Hair ROI: define region above forehead (from top of forehead up to top of head/frame)
        # Bounding box width centered on forehead, extending ~25% face height upwards
        # Estimate face height from chin (152) to top forehead (10)
        chin = landmarks_3d[152, :2]
        face_height = max(abs(chin[1] - top_forehead[1]), 50.0)

        roi_height = int(face_height * 0.40)
        roi_width = int(face_height * 0.70)

        y1 = max(0, fy - roi_height)
        y2 = max(0, fy - int(face_height * 0.05)) # slightly above hairline
        x1 = max(0, fx - roi_width // 2)
        x2 = min(w, fx + roi_width // 2)

        if y2 <= y1 or x2 <= x1:
            return {
                "hair_detected": False,
                "hair_density_score": 50.0,
                "hairline_stability": "Partially Visible",
                "hair_texture_coarseness": 0.0,
                "graying_ratio_percentage": 0.0,
                "summary": "Hair region partially outside frame boundaries."
            }

        hair_crop = frame[y1:y2, x1:x2]
        if hair_crop.size == 0 or hair_crop.shape[0] < 10 or hair_crop.shape[1] < 10:
            return {
                "hair_detected": False,
                "hair_density_score": 50.0,
                "hairline_stability": "Unknown",
                "hair_texture_coarseness": 0.0,
                "graying_ratio_percentage": 0.0,
                "summary": "Hair region crop insufficient for analysis."
            }

        gray_hair = cv2.cvtColor(hair_crop, cv2.COLOR_BGR2GRAY)

        # 1. Edge & Strand Density (Sobel gradients capture fine strand structure)
        sobelx = cv2.Sobel(gray_hair, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_hair, cv2.CV_64F, 0, 1, ksize=3)
        edge_energy = np.mean(np.sqrt(sobelx**2 + sobely**2))

        # Normalize edge energy to a density score 0-100
        density_score = float(np.clip((edge_energy / 45.0) * 100.0, 10.0, 100.0))

        # 2. Texture Coarseness (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray_hair, cv2.CV_64F).var())

        # 3. Graying Ratio (High luminance V > 170, low saturation S < 40 in HSV space)
        hsv_hair = cv2.cvtColor(hair_crop, cv2.COLOR_BGR2HSV)
        sat = hsv_hair[:, :, 1]
        val = hsv_hair[:, :, 2]
        gray_mask = (val > 165) & (sat < 45)
        graying_ratio = (np.count_nonzero(gray_mask) / float(gray_hair.size)) * 100.0

        # 4. Hairline Stability Assessment
        # Contrast between forehead skin tone and hair region
        forehead_y1 = min(h - 1, fy)
        forehead_y2 = min(h, fy + int(face_height * 0.15))
        forehead_crop = frame[forehead_y1:forehead_y2, x1:x2]
        
        hair_detected = True
        if forehead_crop.size > 0:
            mean_fh_l = np.mean(cv2.cvtColor(forehead_crop, cv2.COLOR_BGR2GRAY))
            mean_hair_l = np.mean(gray_hair)
            contrast = abs(mean_fh_l - mean_hair_l)
            if contrast < 12.0 and density_score < 30.0:
                hairline_stability = "Receded / Sparse Hairline"
            else:
                hairline_stability = "Dense / Intact Hairline"
        else:
            hairline_stability = "Normal Hairline"

        summary = f"Hair density score: {density_score:.1f}%. {hairline_stability}. Estimated graying index: {graying_ratio:.1f}%."

        return {
            "hair_detected": hair_detected,
            "hair_density_score": round(density_score, 1),
            "hairline_stability": hairline_stability,
            "hair_texture_coarseness": round(lap_var, 1),
            "graying_ratio_percentage": round(float(graying_ratio), 1),
            "summary": summary
        }

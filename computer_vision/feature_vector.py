from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional

from computer_vision.quality.camera_quality import CameraQualityEngine
from computer_vision.face.landmarks import FaceLandmarkExtractor
from computer_vision.eyes.eye_analyzer import EyeAnalyzer
from computer_vision.skin.skin_analyzer import SkinRegionAnalyzer
from computer_vision.symmetry.symmetry_analyzer import FacialSymmetryAnalyzer
from computer_vision.hair.hair_analyzer import HairAnalyzer

class VisualFeatureExtractor:
    """
    Unified Computer Vision Feature Extractor.
    Executes frame quality checks, face landmark detection, eye analysis (EAR, dark circles, tiredness),
    skin analysis (color, texture, blemishes/acne, erythema, condition patterns),
    hair analysis (density, hairline transition), and bilateral symmetry calculation.
    """

    def __init__(self, model_path: str = "models/face_landmarker.task"):
        self.quality_engine = CameraQualityEngine()
        self.landmark_extractor = FaceLandmarkExtractor(model_path=model_path)
        self.eye_analyzer = EyeAnalyzer()
        self.skin_analyzer = SkinRegionAnalyzer()
        self.symmetry_analyzer = FacialSymmetryAnalyzer()
        self.hair_analyzer = HairAnalyzer()

    def extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a BGR image frame and outputs a versioned structured visual feature dictionary.
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        if frame is None or frame.size == 0:
            return {
                "schema_version": "1.0",
                "timestamp": timestamp,
                "status": "invalid_frame",
                "quality": self.quality_engine.evaluate_frame(None),
                "face": {"face_detected": False, "face_count": 0, "bbox": None},
                "eyes": self.eye_analyzer.analyze_eyes(None, None),
                "symmetry": self.symmetry_analyzer.analyze_symmetry(None),
                "skin": self.skin_analyzer.analyze_skin(frame, None),
                "hair": self.hair_analyzer.analyze_hair(None, None)
            }

        # 1. Landmark extraction
        face_data = self.landmark_extractor.extract_landmarks(frame)

        # 2. Quality evaluation
        quality_data = self.quality_engine.evaluate_frame(frame, face_bbox=face_data["bbox"])

        # 3. If no face detected, return quality + empty feature vectors
        if not face_data["face_detected"]:
            return {
                "schema_version": "1.0",
                "timestamp": timestamp,
                "status": "no_face_detected",
                "quality": quality_data,
                "face": {"face_detected": False, "face_count": 0, "bbox": None},
                "eyes": self.eye_analyzer.analyze_eyes(None, None),
                "symmetry": self.symmetry_analyzer.analyze_symmetry(None),
                "skin": self.skin_analyzer.analyze_skin(frame, None),
                "hair": self.hair_analyzer.analyze_hair(None, None)
            }

        landmarks_3d = face_data["landmarks"]

        # 4. Feature modules with live frame pixel evaluation
        eye_data = self.eye_analyzer.analyze_eyes(landmarks_3d, frame=frame)
        skin_data = self.skin_analyzer.analyze_skin(frame, landmarks_3d)
        symmetry_data = self.symmetry_analyzer.analyze_symmetry(landmarks_3d)
        hair_data = self.hair_analyzer.analyze_hair(frame, landmarks_3d)

        status = "success" if quality_data["overall_quality"] >= 65.0 else "low_quality_success"

        return {
            "schema_version": "1.0",
            "timestamp": timestamp,
            "status": status,
            "quality": quality_data,
            "face": {
                "face_detected": True,
                "face_count": face_data["face_count"],
                "bbox": face_data["bbox"]
            },
            "eyes": eye_data,
            "symmetry": symmetry_data,
            "skin": skin_data,
            "hair": hair_data
        }

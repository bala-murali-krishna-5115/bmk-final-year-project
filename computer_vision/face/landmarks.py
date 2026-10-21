import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from typing import Dict, Any, List, Optional, Tuple

# Key Landmark Indices for MediaPipe Face Mesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
FOREHEAD_INDICES = [10, 338, 297, 332, 284, 109, 67, 103, 54, 68]
LEFT_CHEEK_INDICES = [50, 187, 207, 205, 36, 142, 100, 116]
RIGHT_CHEEK_INDICES = [280, 411, 427, 425, 266, 371, 329, 345]
MIDLINE_INDICES = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

class FaceLandmarkExtractor:
    """
    Extracts 468 3D facial landmarks using MediaPipe 1.0 FaceLandmarker.
    Provides region landmark extraction for eyes, cheeks, forehead, and symmetry computation.
    """

    def __init__(self, model_path: str = "models/face_landmarker.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe task model not found at {model_path}")

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract_landmarks(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes BGR frame and extracts normalized 3D landmarks.

        Returns:
            Dict containing face_detected (bool), face_count (int), bbox, landmarks (N x 3 np.ndarray)
        """
        if frame is None or frame.size == 0:
            return {"face_detected": False, "face_count": 0, "bbox": None, "landmarks": None}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)
        face_count = len(result.face_landmarks) if result.face_landmarks else 0

        if face_count == 0:
            return {"face_detected": False, "face_count": 0, "bbox": None, "landmarks": None}

        # Deterministic selection: Pick primary face (first face)
        landmarks_obj = result.face_landmarks[0]
        h, w, _ = frame.shape
        coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks_obj])

        # Compute bounding box
        min_x, min_y = np.min(coords[:, 0]), np.min(coords[:, 1])
        max_x, max_y = np.max(coords[:, 0]), np.max(coords[:, 1])
        bbox = {
            "x": int(min_x),
            "y": int(min_y),
            "width": int(max_x - min_x),
            "height": int(max_y - min_y)
        }

        return {
            "face_detected": True,
            "face_count": face_count,
            "bbox": bbox,
            "landmarks": coords,
            "normalized_landmarks": np.array([[lm.x, lm.y, lm.z] for lm in landmarks_obj])
        }

    @staticmethod
    def get_region_polygon(coords: np.ndarray, indices: List[int]) -> np.ndarray:
        """Helper to extract 2D integer polygon for landmark indices."""
        return coords[indices, :2].astype(np.int32)

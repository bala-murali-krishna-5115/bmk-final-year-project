import os
import pytest
import numpy as np
import cv2
from computer_vision.feature_vector import VisualFeatureExtractor
from computer_vision.quality.camera_quality import CameraQualityEngine
from computer_vision.eyes.eye_analyzer import EyeAnalyzer

def test_camera_quality_engine():
    """Test quality engine evaluation on synthetic blank and textured frames."""
    engine = CameraQualityEngine()
    
    # Test blank frame
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    q_blank = engine.evaluate_frame(blank_frame)
    assert q_blank["brightness_score"] < 50.0
    assert q_blank["overall_quality"] < 65.0

    # Test bright textured frame
    bright_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    q_bright = engine.evaluate_frame(bright_frame)
    assert q_bright["brightness_mean"] > 100.0

def test_eye_analyzer_ear():
    """Test Eye Aspect Ratio calculation on dummy landmark coordinates."""
    analyzer = EyeAnalyzer()

    # Dummy landmarks for open eye (high vertical distance)
    dummy_landmarks = np.zeros((468, 3), dtype=np.float32)
    # Left eye: 33 (left), 133 (right) horizontal distance = 100
    dummy_landmarks[33] = [100, 200, 0]
    dummy_landmarks[133] = [200, 200, 0]
    # Vertical points: 160-144 = 40, 158-153 = 40
    dummy_landmarks[160] = [133, 180, 0]
    dummy_landmarks[144] = [133, 220, 0]
    dummy_landmarks[158] = [166, 180, 0]
    dummy_landmarks[153] = [166, 220, 0]

    result = analyzer.analyze_eyes(dummy_landmarks)
    assert result["left_eye_openness"] > 0.35
    assert result["eye_state"] == "open"

def test_visual_feature_extractor_no_face():
    """Test VisualFeatureExtractor on a frame with no face."""
    extractor = VisualFeatureExtractor(model_path="models/face_landmarker.task")
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)

    features = extractor.extract_features(blank)
    assert features["schema_version"] == "1.0"
    assert features["status"] == "no_face_detected"
    assert features["face"]["face_detected"] is False
    assert "eyes" in features
    assert "skin" in features
    assert "symmetry" in features

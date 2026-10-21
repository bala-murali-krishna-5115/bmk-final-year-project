import pytest
import numpy as np

from computer_vision.eyes.eye_analyzer import EyeAnalyzer
from computer_vision.skin.skin_analyzer import SkinRegionAnalyzer
from computer_vision.hair.hair_analyzer import HairAnalyzer
from backend.services.recommendation_service import RecommendationService

def test_eye_analyzer_dark_circles_and_tiredness():
    """Verify EyeAnalyzer computes under-eye dark circles and tiredness score."""
    analyzer = EyeAnalyzer()

    # Synthetic 468 landmark coordinates (468, 3)
    coords = np.zeros((468, 3), dtype=np.float32)
    for i in range(468):
        coords[i] = [100.0 + (i % 20) * 10, 100.0 + (i // 20) * 10, 0.0]

    # Synthetic BGR frame
    frame = np.full((400, 400, 3), 150, dtype=np.uint8)

    res = analyzer.analyze_eyes(coords, frame=frame)
    assert "dark_circles" in res
    assert "tiredness" in res
    assert "severity" in res["dark_circles"]
    assert "darkness_index_percentage" in res["dark_circles"]
    assert "tiredness_score" in res["tiredness"]
    assert "tiredness_level" in res["tiredness"]

def test_skin_analyzer_blemishes_and_pigmentation():
    """Verify SkinRegionAnalyzer detects blemishes, erythema, and screens condition patterns."""
    analyzer = SkinRegionAnalyzer()

    coords = np.zeros((468, 3), dtype=np.float32)
    for i in range(468):
        coords[i] = [100.0 + (i % 20) * 10, 100.0 + (i // 20) * 10, 0.0]

    frame = np.full((400, 400, 3), 140, dtype=np.uint8)
    # Add a reddish blemish spot
    frame[150:156, 150:156] = [60, 60, 220]

    skin_res = analyzer.analyze_skin(frame, coords)
    assert "blemishes" in skin_res
    assert "pigmentation" in skin_res
    assert "condition_pattern" in skin_res
    assert "visible_lesion_count" in skin_res["blemishes"]
    assert "erythema_index" in skin_res["pigmentation"]
    assert "SCIN" in skin_res["condition_pattern"]["reference_datasets"]
    assert "Fitzpatrick17k" in skin_res["condition_pattern"]["reference_datasets"]

def test_hair_analyzer():
    """Verify HairAnalyzer extracts hairline stability and density."""
    analyzer = HairAnalyzer()
    coords = np.zeros((468, 3), dtype=np.float32)
    coords[10] = [200.0, 150.0, 0.0]  # top of forehead
    coords[152] = [200.0, 350.0, 0.0] # chin

    frame = np.full((400, 400, 3), 50, dtype=np.uint8)
    res = analyzer.analyze_hair(frame, coords)
    assert "hair_detected" in res
    assert "hair_density_score" in res
    assert "hairline_stability" in res

def test_recommendation_service_no_fake_hardware_and_rest_advice():
    """Verify RecommendationService omits fake vitals when hardware is missing and gives resting guidance."""
    # Case 1: Hardware Missing & Dark Circles Present
    vis = {
        "eyes": {
            "dark_circles": {
                "darkness_index_percentage": 14.5,
                "severity": "Moderate"
            },
            "tiredness": {
                "tiredness_score": 65.0,
                "tiredness_level": "Fatigued / Sleep Deprived"
            }
        },
        "skin": {
            "blemishes": {"visible_lesion_count": 4, "blemish_severity": "Mild"},
            "pigmentation": {"erythema_index": 14.0, "redness_flag": False}
        }
    }

    report = RecommendationService.generate_recommendations(
        visual_features=vis,
        sensor_readings=None,
        hardware_connected=False
    )

    assert report["hardware_connected"] is False
    assert "DISCONNECTED" in report["hardware_status"].upper()
    assert "You need to take rest" in report["executive_summary"]
    assert any("take rest" in rec.lower() for rec in report["specific_recommendations"])
    assert "NOT a 100% prescribed" in report["disclaimer"]

    # Case 2: Hardware Connected
    sensors = {"heart_rate": 75.0, "spo2": 98.5, "temperature": 36.6}
    report_hw = RecommendationService.generate_recommendations(
        visual_features=vis,
        sensor_readings=sensors,
        hardware_connected=True
    )
    assert report_hw["hardware_connected"] is True
    assert "Connected" in report_hw["hardware_status"]
    assert "75.0 bpm" in report_hw["sensor_message"]

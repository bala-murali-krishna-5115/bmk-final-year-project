from typing import Dict, Any, Optional, Tuple, List

class QualityService:
    """
    Evaluates independent and combined data quality across Computer Vision and Physiological Sensors.
    Enforces the mandatory 65% minimum quality threshold for reliable screening inference.
    """

    MIN_QUALITY_THRESHOLD: float = 65.0

    @classmethod
    def evaluate_sensor_quality(cls, sensor_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Validates physiological sensor measurements against physical physiological limits.
        Computes sensor quality score (0.0 to 100.0) and generates diagnostic warnings.
        """
        score = 100.0
        warnings = []

        if not sensor_data:
            return 0.0, ["No physiological sensor readings provided."]

        hr = sensor_data.get("heart_rate")
        spo2 = sensor_data.get("spo2")
        temp = sensor_data.get("temperature")

        # Heart rate validation (Nominal resting: 50-120 bpm, Hard limits: 40-200 bpm)
        if hr is None:
            score -= 30.0
            warnings.append("Missing heart rate measurement.")
        elif hr < 40.0 or hr > 200.0:
            score -= 40.0
            warnings.append(f"Heart rate ({hr} bpm) out of physiological bounds [40, 200]. Possible sensor contact loss.")
        elif hr < 50.0 or hr > 130.0:
            score -= 10.0
            warnings.append(f"Heart rate ({hr} bpm) shows extreme variation.")

        # SpO2 validation (Nominal: 94-100%, Hard limits: 70-100%)
        if spo2 is None:
            score -= 35.0
            warnings.append("Missing SpO2 blood oxygen saturation measurement.")
        elif spo2 < 70.0 or spo2 > 100.0:
            score -= 40.0
            warnings.append(f"SpO2 ({spo2}%) out of physiological bounds [70, 100]. Sensor displacement or severe artifact.")
        elif spo2 < 90.0:
            score -= 10.0
            warnings.append(f"Low SpO2 reading ({spo2}%). Verify probe placement.")

        # Temperature validation (Nominal: 35.5-38.0°C, Hard limits: 30-43°C)
        if temp is None:
            score -= 25.0
            warnings.append("Missing body temperature measurement.")
        elif temp < 30.0 or temp > 43.0:
            score -= 35.0
            warnings.append(f"Temperature ({temp}°C) out of physical bounds [30.0, 43.0]. Check thermal ambient probe.")
        elif temp < 35.0 or temp > 39.5:
            score -= 10.0
            warnings.append(f"Atypical thermal reading ({temp}°C).")

        sensor_score = max(0.0, min(100.0, round(score, 1)))
        return sensor_score, warnings

    @classmethod
    def evaluate_overall_quality(
        cls,
        camera_quality_score: Optional[float],
        sensor_quality_score: Optional[float],
        has_camera: bool = True,
        has_sensor: bool = True
    ) -> Dict[str, Any]:
        """
        Combines camera and sensor quality scores into unified screening quality metric.
        Applies refusal rule if overall_data_quality < 65%.
        """
        cam_score = float(camera_quality_score) if camera_quality_score is not None else 0.0
        sens_score = float(sensor_quality_score) if sensor_quality_score is not None else 0.0

        if has_camera and has_sensor:
            overall = 0.5 * cam_score + 0.5 * sens_score
        elif has_camera:
            overall = cam_score
        elif has_sensor:
            overall = sens_score
        else:
            overall = 0.0

        overall = round(max(0.0, min(100.0, overall)), 1)
        quality_pass = overall >= cls.MIN_QUALITY_THRESHOLD

        rejection_reason = None
        if not quality_pass:
            reasons = []
            if cam_score < cls.MIN_QUALITY_THRESHOLD and has_camera:
                reasons.append(f"Camera quality score ({cam_score}%) is below minimum threshold ({cls.MIN_QUALITY_THRESHOLD}%)")
            if sens_score < cls.MIN_QUALITY_THRESHOLD and has_sensor:
                reasons.append(f"Sensor quality score ({sens_score}%) is below minimum threshold ({cls.MIN_QUALITY_THRESHOLD}%)")
            if not reasons:
                reasons.append(f"Combined quality score ({overall}%) is below minimum threshold ({cls.MIN_QUALITY_THRESHOLD}%)")
            rejection_reason = "; ".join(reasons)

        return {
            "camera_quality_score": cam_score,
            "sensor_quality_score": sens_score,
            "overall_data_quality": overall,
            "quality_pass": quality_pass,
            "rejection_reason": rejection_reason,
            "threshold": cls.MIN_QUALITY_THRESHOLD
        }

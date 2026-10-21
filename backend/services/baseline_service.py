import json
import logging
import numpy as np
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.models.database_models import User, BaselineProfile, ScreeningSession, VisualFeature, SensorReading

logger = logging.getLogger(__name__)

class BaselineService:
    """
    Manages individualized baseline profiles and anomaly distances.
    Learns normal baseline distributions for specific subjects over time.
    """

    @classmethod
    def get_or_create_user(cls, db: Session, user_code: str) -> User:
        user = db.query(User).filter(User.user_code == user_code).first()
        if not user:
            user = User(user_code=user_code)
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    @classmethod
    def get_user_baseline(cls, db: Session, user_code: str) -> Optional[Dict[str, Any]]:
        user = db.query(User).filter(User.user_code == user_code).first()
        if not user or not user.baseline:
            return None
        try:
            return json.loads(user.baseline.baseline_vector_json)
        except Exception:
            return None

    @classmethod
    def update_user_baseline(cls, db: Session, user_code: str, custom_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Updates or aggregates user baseline profile.
        If custom_profile is None, computes mean and standard deviations from historical completed sessions.
        """
        user = cls.get_or_create_user(db, user_code)

        if custom_profile:
            profile_data = custom_profile
        else:
            # Query historical sessions for this user
            sessions = db.query(ScreeningSession).filter(
                ScreeningSession.user_id == user.id,
                ScreeningSession.status == "completed"
            ).all()

            if not sessions:
                # Default baseline profile
                profile_data = {
                    "means": {
                        "heart_rate": 72.0,
                        "spo2": 98.2,
                        "temperature": 36.6,
                        "facial_symmetry_score": 0.92,
                        "mean_openness": 0.28
                    },
                    "stds": {
                        "heart_rate": 6.0,
                        "spo2": 1.0,
                        "temperature": 0.3,
                        "facial_symmetry_score": 0.04,
                        "mean_openness": 0.03
                    },
                    "sample_count": 0
                }
            else:
                hr_list, spo2_list, temp_list, sym_list, ear_list = [], [], [], [], []
                for s in sessions:
                    for sr in s.sensor_readings:
                        if sr.heart_rate: hr_list.append(sr.heart_rate)
                        if sr.spo2: spo2_list.append(sr.spo2)
                        if sr.temperature: temp_list.append(sr.temperature)
                    for vf in s.visual_features:
                        if vf.facial_symmetry_score: sym_list.append(vf.facial_symmetry_score)
                        if vf.eye_openness: ear_list.append(vf.eye_openness)

                profile_data = {
                    "means": {
                        "heart_rate": float(np.mean(hr_list)) if hr_list else 72.0,
                        "spo2": float(np.mean(spo2_list)) if spo2_list else 98.2,
                        "temperature": float(np.mean(temp_list)) if temp_list else 36.6,
                        "facial_symmetry_score": float(np.mean(sym_list)) if sym_list else 0.92,
                        "mean_openness": float(np.mean(ear_list)) if ear_list else 0.28
                    },
                    "stds": {
                        "heart_rate": float(max(np.std(hr_list), 3.0)) if hr_list else 6.0,
                        "spo2": float(max(np.std(spo2_list), 0.5)) if spo2_list else 1.0,
                        "temperature": float(max(np.std(temp_list), 0.2)) if temp_list else 0.3,
                        "facial_symmetry_score": float(max(np.std(sym_list), 0.02)) if sym_list else 0.04,
                        "mean_openness": float(max(np.std(ear_list), 0.02)) if ear_list else 0.03
                    },
                    "sample_count": len(sessions)
                }

        json_str = json.dumps(profile_data)
        if user.baseline:
            user.baseline.baseline_vector_json = json_str
        else:
            baseline_record = BaselineProfile(user_id=user.id, baseline_vector_json=json_str)
            db.add(baseline_record)
        db.commit()

        return profile_data

    @classmethod
    def calculate_deviation(cls, current_features: Dict[str, float], baseline_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes normalized Z-scores and Mahalanobis/Euclidean distance from baseline profile.
        """
        if not baseline_profile or "means" not in baseline_profile or "stds" not in baseline_profile:
            return {
                "has_baseline": False,
                "baseline_distance": 0.0,
                "deviation_flag": False,
                "z_scores": {}
            }

        means = baseline_profile["means"]
        stds = baseline_profile["stds"]

        z_scores = {}
        sum_sq = 0.0
        count = 0

        for key, mean_val in means.items():
            if key in current_features and current_features[key] is not None:
                val = float(current_features[key])
                std_val = float(stds.get(key, 1.0))
                std_val = max(std_val, 1e-4)
                z = (val - mean_val) / std_val
                z_scores[key] = round(z, 2)
                sum_sq += z ** 2
                count += 1

        dist = round(float(np.sqrt(sum_sq / max(count, 1))), 2)
        flag = dist >= 2.0

        return {
            "has_baseline": True,
            "baseline_distance": dist,
            "deviation_flag": flag,
            "z_scores": z_scores,
            "sample_count": baseline_profile.get("sample_count", 0)
        }

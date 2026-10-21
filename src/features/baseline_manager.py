import numpy as np
from typing import Dict, Any, List, Optional

class PersonalBaselineManager:
    """
    Computes personalized feature baselines (mean, std per user) and calculates
    feature deltas and normalized deviation scores against established baselines.
    """

    @staticmethod
    def compute_baseline_profile(feature_matrix: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Calculates baseline mean and std for a user across multiple historical sessions.
        """
        if feature_matrix is None or len(feature_matrix) == 0:
            return {}

        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)

        profile = {
            "num_sessions": len(feature_matrix),
            "means": {name: float(m) for name, m in zip(feature_names, means)},
            "stds": {name: float(s) for name, s in zip(feature_names, stds)}
        }
        return profile

    @staticmethod
    def calculate_baseline_deviation(
        current_features: Dict[str, float],
        baseline_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates normalized Mahalanobis/Z-score distance and per-feature deltas
        between current observation and established user baseline.
        """
        if not baseline_profile or "means" not in baseline_profile:
            return {
                "baseline_distance": 0.0,
                "deviation_flag": False,
                "status": "no_baseline_established",
                "feature_deltas": {}
            }

        means = baseline_profile["means"]
        stds = baseline_profile["stds"]

        z_scores = []
        deltas = {}

        for feat, val in current_features.items():
            if feat in means:
                mu = means[feat]
                sigma = max(stds[feat], 1e-4)
                delta = val - mu
                z_score = delta / sigma
                deltas[feat] = {
                    "current": round(float(val), 4),
                    "baseline_mean": round(float(mu), 4),
                    "delta": round(float(delta), 4),
                    "z_score": round(float(z_score), 4)
                }
                z_scores.append(z_score ** 2)

        # Mahalanobis / Normalized Euclidean Z-score distance
        baseline_distance = float(np.sqrt(np.mean(z_scores))) if z_scores else 0.0
        deviation_flag = baseline_distance >= 2.0  # > 2 standard deviations away

        return {
            "baseline_distance": round(baseline_distance, 4),
            "deviation_flag": deviation_flag,
            "status": "within_learned_range" if not deviation_flag else "deviation_detected",
            "feature_deltas": deltas
        }

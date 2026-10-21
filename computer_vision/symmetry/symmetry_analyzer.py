import numpy as np
from typing import Dict, Any, Optional, Tuple
from computer_vision.face.landmarks import MIDLINE_INDICES

class FacialSymmetryAnalyzer:
    """
    Computes landmark-based bilateral facial symmetry score.
    Validates head pose orientation to prevent skewed symmetry ratings.
    """

    @staticmethod
    def _estimate_head_pose(landmarks_3d: np.ndarray) -> Tuple[float, float, float]:
        """Estimates approximate pitch, yaw, roll in degrees from 3D landmarks."""
        nose_tip = landmarks_3d[1]
        chin = landmarks_3d[152]
        left_eye = landmarks_3d[33]
        right_eye = landmarks_3d[263]

        # Roll: angle between eyes
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        roll = np.degrees(np.arctan2(eye_dy, eye_dx))

        # Yaw: relative distance of nose tip between left and right eyes
        dist_left = np.linalg.norm(nose_tip[:2] - left_eye[:2])
        dist_right = np.linalg.norm(nose_tip[:2] - right_eye[:2])
        yaw = float(abs(dist_left - dist_right) / max(dist_left + dist_right, 1e-5) * 90.0)

        # Pitch: nose to chin vs eye to nose vertical ratio
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        pitch = float(abs((nose_tip[1] - eye_mid_y) - (chin[1] - nose_tip[1])) / max(chin[1] - eye_mid_y, 1e-5) * 45.0)

        return pitch, yaw, roll

    def analyze_symmetry(self, landmarks_3d: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Calculates bilateral facial symmetry score ($0.0$ to $1.0$).
        """
        if landmarks_3d is None or len(landmarks_3d) < 468:
            return {
                "facial_symmetry_score": 0.0,
                "left_right_difference": 0.0,
                "valid_for_symmetry": False,
                "pose_angles": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
            }

        pitch, yaw, roll = self._estimate_head_pose(landmarks_3d)

        # Check pose limits (Max 15 degrees rotation allowed)
        if abs(pitch) > 20.0 or abs(yaw) > 15.0 or abs(roll) > 15.0:
            return {
                "facial_symmetry_score": 0.0,
                "left_right_difference": round(float(abs(yaw)), 2),
                "valid_for_symmetry": False,
                "pose_angles": {"pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1)}
            }

        # Bilateral landmark pairs (Left, Right)
        pairs = [(33, 263), (61, 291), (234, 454), (130, 359), (50, 280)]
        midline_pt = np.mean(landmarks_3d[MIDLINE_INDICES, :2], axis=0)

        diffs = []
        for left_idx, right_idx in pairs:
            dist_left = np.linalg.norm(landmarks_3d[left_idx, :2] - midline_pt)
            dist_right = np.linalg.norm(landmarks_3d[right_idx, :2] - midline_pt)
            max_d = max(dist_left, dist_right, 1e-5)
            diffs.append(abs(dist_left - dist_right) / max_d)

        mean_diff = float(np.mean(diffs))
        symmetry_score = float(np.clip(1.0 - mean_diff, 0.0, 1.0))

        return {
            "facial_symmetry_score": round(symmetry_score, 4),
            "left_right_difference": round(mean_diff, 4),
            "valid_for_symmetry": True,
            "pose_angles": {"pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1)}
        }

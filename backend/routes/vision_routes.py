import base64
import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from datetime import datetime

from backend.database.connection import SessionLocal
from backend.models.database_models import ScreeningSession, VisualFeature
from computer_vision.feature_vector import VisualFeatureExtractor

vision_bp = Blueprint('vision_bp', __name__, url_prefix='/api/vision')

# Singleton visual feature extractor
feature_extractor = None

def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = VisualFeatureExtractor()
    return feature_extractor

@vision_bp.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """
    Accepts a base64-encoded webcam image frame from the browser frontend,
    executes facial landmark extraction, camera quality checks, eye analysis,
    facial symmetry, and skin color/texture analysis.
    """
    data = request.get_json() or {}
    session_id = data.get('session_id')
    frame_b64 = data.get('frame_base64') or data.get('image')

    if not session_id or not frame_b64:
        return jsonify({"error": "Missing required fields: session_id and frame_base64"}), 400

    # Strip data URI prefix if present (e.g. data:image/jpeg;base64,...)
    if ',' in frame_b64:
        frame_b64 = frame_b64.split(',', 1)[1]

    try:
        image_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode base64 image frame"}), 400
    except Exception as e:
        return jsonify({"error": f"Image decoding error: {str(e)}"}), 400

    extractor = get_feature_extractor()
    features = extractor.extract_features(frame)

    # Save to database
    db = SessionLocal()
    try:
        session = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
        if not session:
            return jsonify({"error": f"ScreeningSession {session_id} not found"}), 404

        quality = features.get("quality", {})
        eyes = features.get("eyes", {})
        symmetry = features.get("symmetry", {})
        skin = features.get("skin", {})

        vf = VisualFeature(
            session_id=session.id,
            ear_left=eyes.get("left_eye_openness"),
            ear_right=eyes.get("right_eye_openness"),
            eye_openness=eyes.get("mean_openness"),
            facial_symmetry_score=symmetry.get("facial_symmetry_score"),
            skin_r_mean=skin.get("skin_r_mean"),
            skin_g_mean=skin.get("skin_g_mean"),
            skin_b_mean=skin.get("skin_b_mean"),
            skin_h_mean=skin.get("skin_h_mean"),
            skin_s_mean=skin.get("skin_s_mean"),
            skin_v_mean=skin.get("skin_v_mean"),
            skin_l_mean=skin.get("skin_l_mean"),
            skin_a_mean=skin.get("skin_a_mean"),
            skin_b_lab_mean=skin.get("skin_b_lab_mean"),
            skin_texture_lbp=skin.get("skin_texture_lbp"),
            camera_quality_score=quality.get("overall_quality", 100.0)
        )
        db.add(vf)
        db.commit()
        db.refresh(vf)

        return jsonify({
            "status": "success",
            "feature_id": vf.id,
            "session_id": session_id,
            "features": features
        }), 200
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@vision_bp.route('/<session_id>/latest', methods=['GET'])
def get_latest_vision(session_id):
    """Retrieves most recent visual feature record for the session."""
    db = SessionLocal()
    try:
        vf = db.query(VisualFeature).filter(
            VisualFeature.session_id == session_id
        ).order_by(VisualFeature.timestamp.desc()).first()

        if not vf:
            return jsonify({"error": "No visual features found for this session"}), 404

        return jsonify({
            "session_id": session_id,
            "ear_left": vf.ear_left,
            "ear_right": vf.ear_right,
            "eye_openness": vf.eye_openness,
            "facial_symmetry_score": vf.facial_symmetry_score,
            "skin_l_mean": vf.skin_l_mean,
            "skin_texture_lbp": vf.skin_texture_lbp,
            "camera_quality_score": vf.camera_quality_score,
            "timestamp": vf.timestamp.isoformat() + "Z"
        }), 200
    finally:
        db.close()

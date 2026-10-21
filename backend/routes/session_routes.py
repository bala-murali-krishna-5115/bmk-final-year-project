from flask import Blueprint, request, jsonify
from datetime import datetime
from backend.database.connection import SessionLocal
from backend.models.database_models import User, ScreeningSession
from backend.services.baseline_service import BaselineService

session_bp = Blueprint('session_bp', __name__, url_prefix='/api/sessions')

@session_bp.route('', methods=['POST'])
def create_session():
    """
    Creates a new screening session for a user (anonymous user_code).
    """
    data = request.get_json() or {}
    user_code = data.get('user_code') or data.get('external_reference') or "ANON_USER"
    has_camera = data.get('has_camera', True)
    has_sensors = data.get('has_sensors', True)

    db = SessionLocal()
    try:
        user = BaselineService.get_or_create_user(db, user_code)
        session = ScreeningSession(
            user_id=user.id,
            status="in_progress",
            has_camera=has_camera,
            has_sensors=has_sensors
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        return jsonify({
            "status": "created",
            "session_id": session.id,
            "user_id": user.id,
            "user_code": user.user_code,
            "created_at": session.timestamp.isoformat() + "Z",
            "has_camera": session.has_camera,
            "has_sensors": session.has_sensors
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@session_bp.route('', methods=['GET'])
def list_sessions():
    """Lists recent screening sessions with user reference."""
    db = SessionLocal()
    try:
        sessions = db.query(ScreeningSession).order_by(ScreeningSession.timestamp.desc()).limit(20).all()
        results = []
        for s in sessions:
            results.append({
                "session_id": s.id,
                "user_code": s.user.user_code if s.user else "Unknown",
                "timestamp": s.timestamp.isoformat() + "Z",
                "status": s.status,
                "has_camera": s.has_camera,
                "has_sensors": s.has_sensors,
                "anomaly_flag": s.anomaly_result.anomaly_flag if s.anomaly_result else False,
                "anomaly_score": s.anomaly_result.anomaly_score if s.anomaly_result else None
            })
        return jsonify({"sessions": results, "count": len(results)}), 200
    finally:
        db.close()

@session_bp.route('/<session_id>', methods=['GET'])
def get_session(session_id):
    """Retrieves session details by session_id."""
    db = SessionLocal()
    try:
        s = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
        if not s:
            return jsonify({"error": "Session not found"}), 404

        return jsonify({
            "session_id": s.id,
            "user_code": s.user.user_code if s.user else "Unknown",
            "timestamp": s.timestamp.isoformat() + "Z",
            "status": s.status,
            "sensor_readings_count": len(s.sensor_readings),
            "visual_features_count": len(s.visual_features),
            "quality_pass": s.data_quality.quality_pass if s.data_quality else None,
            "anomaly_score": s.anomaly_result.anomaly_score if s.anomaly_result else None
        }), 200
    finally:
        db.close()

@session_bp.route('/<session_id>/complete', methods=['POST'])
def complete_session(session_id):
    """Marks session status as completed."""
    db = SessionLocal()
    try:
        s = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
        if not s:
            return jsonify({"error": "Session not found"}), 404

        s.status = "completed"
        db.commit()
        return jsonify({"status": "completed", "session_id": s.id}), 200
    finally:
        db.close()

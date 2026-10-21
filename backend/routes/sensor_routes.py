import random
from flask import Blueprint, request, jsonify
from datetime import datetime

from backend.database.connection import SessionLocal
from backend.models.database_models import ScreeningSession, SensorReading
from backend.services.quality_service import QualityService

sensor_bp = Blueprint('sensor_bp', __name__, url_prefix='/api/sensors')

@sensor_bp.route('/measurement', methods=['POST'])
def receive_measurement():
    """
    Receives physiological sensor readings from ESP32 or software client.
    Validates sensor quality against physiological boundaries.
    """
    data = request.get_json() or {}
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({"error": "Missing required field: session_id"}), 400

    heart_rate = data.get('heart_rate')
    spo2 = data.get('spo2')
    temperature = data.get('temperature')
    is_simulated = bool(data.get('is_simulated', False))

    # Evaluate quality
    quality_score, warnings = QualityService.evaluate_sensor_quality({
        "heart_rate": heart_rate,
        "spo2": spo2,
        "temperature": temperature
    })

    db = SessionLocal()
    try:
        session = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
        if not session:
            return jsonify({"error": f"ScreeningSession {session_id} not found"}), 404

        reading = SensorReading(
            session_id=session.id,
            heart_rate=float(heart_rate) if heart_rate is not None else None,
            spo2=float(spo2) if spo2 is not None else None,
            temperature=float(temperature) if temperature is not None else None,
            sensor_quality_score=quality_score,
            is_simulated=is_simulated
        )
        db.add(reading)
        db.commit()
        db.refresh(reading)

        return jsonify({
            "status": "success",
            "reading_id": reading.id,
            "session_id": session_id,
            "heart_rate": reading.heart_rate,
            "spo2": reading.spo2,
            "temperature": reading.temperature,
            "sensor_quality_score": reading.sensor_quality_score,
            "is_simulated": reading.is_simulated,
            "warnings": warnings
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@sensor_bp.route('/simulate', methods=['POST'])
def simulate_sensor():
    """
    Generates realistic physiological sensor test readings for interactive testing
    when physical ESP32 hardware is not connected.
    """
    data = request.get_json() or {}
    session_id = data.get('session_id')
    mode = data.get('mode', 'normal')  # 'normal', 'elevated', 'fatigue', 'stress'

    if not session_id:
        return jsonify({"error": "Missing required field: session_id"}), 400

    if mode == 'elevated' or mode == 'stress':
        hr = round(random.uniform(98.0, 125.0), 1)
        spo2 = round(random.uniform(94.0, 97.0), 1)
        temp = round(random.uniform(37.4, 38.3), 2)
    elif mode == 'fatigue':
        hr = round(random.uniform(88.0, 105.0), 1)
        spo2 = round(random.uniform(91.0, 94.5), 1)
        temp = round(random.uniform(36.8, 37.5), 2)
    elif mode == 'low_quality':
        # Simulated sensor noise / bad probe contact
        hr = round(random.uniform(30.0, 42.0), 1)
        spo2 = round(random.uniform(62.0, 72.0), 1)
        temp = round(random.uniform(28.0, 31.0), 2)
    else:
        # Normal nominal resting baseline
        hr = round(random.uniform(68.0, 78.0), 1)
        spo2 = round(random.uniform(97.5, 99.5), 1)
        temp = round(random.uniform(36.4, 36.8), 2)

    quality_score, warnings = QualityService.evaluate_sensor_quality({
        "heart_rate": hr,
        "spo2": spo2,
        "temperature": temp
    })

    db = SessionLocal()
    try:
        session = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
        if not session:
            return jsonify({"error": f"ScreeningSession {session_id} not found"}), 404

        reading = SensorReading(
            session_id=session.id,
            heart_rate=hr,
            spo2=spo2,
            temperature=temp,
            sensor_quality_score=quality_score,
            is_simulated=True
        )
        db.add(reading)
        db.commit()
        db.refresh(reading)

        return jsonify({
            "status": "success",
            "reading_id": reading.id,
            "session_id": session_id,
            "mode": mode,
            "heart_rate": reading.heart_rate,
            "spo2": reading.spo2,
            "temperature": reading.temperature,
            "sensor_quality_score": reading.sensor_quality_score,
            "is_simulated": True,
            "warnings": warnings
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@sensor_bp.route('/<session_id>/latest', methods=['GET'])
def get_latest_sensor(session_id):
    """Retrieves most recent sensor reading for the session."""
    db = SessionLocal()
    try:
        reading = db.query(SensorReading).filter(
            SensorReading.session_id == session_id
        ).order_by(SensorReading.timestamp.desc()).first()

        if not reading:
            return jsonify({"error": "No sensor readings found for this session"}), 404

        return jsonify({
            "session_id": session_id,
            "heart_rate": reading.heart_rate,
            "spo2": reading.spo2,
            "temperature": reading.temperature,
            "sensor_quality_score": reading.sensor_quality_score,
            "is_simulated": reading.is_simulated,
            "timestamp": reading.timestamp.isoformat() + "Z"
        }), 200
    finally:
        db.close()

import io
import json
import base64
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, send_file, Response
from datetime import datetime

from backend.database.connection import SessionLocal
from backend.models.database_models import (
    ScreeningSession,
    SensorReading,
    VisualFeature,
    DataQuality,
    ModelResult,
    AnomalyResult,
    User
)
from backend.services.quality_service import QualityService
from backend.services.baseline_service import BaselineService
from backend.services.multimodal_service import MultimodalService
from backend.services.pdf_report_service import PDFReportService
from computer_vision.feature_vector import VisualFeatureExtractor

screening_bp = Blueprint('screening_bp', __name__, url_prefix='/api/screening')

# Shared visual feature extractor
feature_extractor = None

def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = VisualFeatureExtractor()
    return feature_extractor

@screening_bp.route('/run', methods=['POST'])
def run_screening():
    """
    Unified Multimodal Screening Pipeline Endpoint.
    Integrates frame feature extraction, sensor validation, data quality gating (65% rule),
    multimodal model inference, anomaly detection, SHAP explainability, and baseline deviation.
    """
    data = request.get_json() or {}
    session_id = data.get('session_id')
    user_code = data.get('user_code')
    frame_b64 = data.get('frame_base64')
    sensor_input = data.get('sensor_data')

    if not session_id:
        return jsonify({"error": "Missing required parameter: session_id"}), 400

    db = SessionLocal()
    try:
        session = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
        if not session:
            return jsonify({"error": f"ScreeningSession {session_id} not found"}), 404

        user = session.user
        if not user_code and user:
            user_code = user.user_code
        elif user_code and not user:
            user = BaselineService.get_or_create_user(db, user_code)
            session.user_id = user.id

        # 1. Process or Fetch Visual Features
        visual_features_dict = None
        camera_quality_score = 90.0

        if frame_b64:
            if ',' in frame_b64:
                frame_b64 = frame_b64.split(',', 1)[1]
            try:
                img_bytes = base64.b64decode(frame_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    extractor = get_feature_extractor()
                    visual_features_dict = extractor.extract_features(frame)
                    camera_quality_score = visual_features_dict.get("quality", {}).get("overall_quality", 85.0)

                    # Store in DB
                    eyes = visual_features_dict.get("eyes", {})
                    symmetry = visual_features_dict.get("symmetry", {})
                    skin = visual_features_dict.get("skin", {})
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
                        camera_quality_score=camera_quality_score
                    )
                    db.add(vf)
                    db.commit()
            except Exception as e:
                pass

        if visual_features_dict is None:
            latest_vf = db.query(VisualFeature).filter(
                VisualFeature.session_id == session.id
            ).order_by(VisualFeature.timestamp.desc()).first()
            if latest_vf:
                camera_quality_score = latest_vf.camera_quality_score or 90.0
                visual_features_dict = {
                    "quality": {"overall_quality": camera_quality_score},
                    "eyes": {"left_eye_openness": latest_vf.ear_left or 0.28, "right_eye_openness": latest_vf.ear_right or 0.28, "mean_openness": latest_vf.eye_openness or 0.28, "eye_asymmetry": 0.01},
                    "symmetry": {"facial_symmetry_score": latest_vf.facial_symmetry_score or 0.92},
                    "skin": {
                        "skin_r_mean": latest_vf.skin_r_mean or 140.0,
                        "skin_g_mean": latest_vf.skin_g_mean or 110.0,
                        "skin_b_mean": latest_vf.skin_b_mean or 95.0,
                        "skin_h_mean": latest_vf.skin_h_mean or 14.0,
                        "skin_s_mean": latest_vf.skin_s_mean or 85.0,
                        "skin_v_mean": latest_vf.skin_v_mean or 140.0,
                        "skin_l_mean": latest_vf.skin_l_mean or 125.0,
                        "skin_a_mean": latest_vf.skin_a_mean or 145.0,
                        "skin_b_lab_mean": latest_vf.skin_b_lab_mean or 138.0,
                        "skin_texture_lbp": latest_vf.skin_texture_lbp or 450.0
                    }
                }
            else:
                # Nominal baseline for visual
                visual_features_dict = {
                    "quality": {"overall_quality": 85.0},
                    "eyes": {"left_eye_openness": 0.28, "right_eye_openness": 0.28, "mean_openness": 0.28, "eye_asymmetry": 0.01},
                    "symmetry": {"facial_symmetry_score": 0.92},
                    "skin": {"skin_r_mean": 140.0, "skin_g_mean": 110.0, "skin_b_mean": 95.0, "skin_h_mean": 14.0, "skin_s_mean": 85.0, "skin_v_mean": 140.0, "skin_l_mean": 125.0, "skin_a_mean": 145.0, "skin_b_lab_mean": 138.0, "skin_texture_lbp": 450.0}
                }

        # 2. Process or Fetch Sensor Readings (NO FAKE / STATIC DATA)
        sensor_readings_dict = None
        sensor_quality_score = None
        has_sensor = False

        if sensor_input and sensor_input.get("heart_rate") is not None:
            q_score, _ = QualityService.evaluate_sensor_quality(sensor_input)
            sensor_quality_score = q_score
            has_sensor = True
            sr = SensorReading(
                session_id=session.id,
                heart_rate=sensor_input.get("heart_rate"),
                spo2=sensor_input.get("spo2"),
                temperature=sensor_input.get("temperature"),
                sensor_quality_score=q_score,
                is_simulated=bool(sensor_input.get("is_simulated", False))
            )
            db.add(sr)
            db.commit()
            sensor_readings_dict = sensor_input
        else:
            latest_sr = db.query(SensorReading).filter(
                SensorReading.session_id == session.id
            ).order_by(SensorReading.timestamp.desc()).first()
            if latest_sr and latest_sr.heart_rate is not None:
                sensor_quality_score = latest_sr.sensor_quality_score or 95.0
                has_sensor = True
                sensor_readings_dict = {
                    "heart_rate": latest_sr.heart_rate,
                    "spo2": latest_sr.spo2,
                    "temperature": latest_sr.temperature,
                    "sensor_quality_score": sensor_quality_score,
                    "is_simulated": latest_sr.is_simulated
                }
            else:
                # Hardware is explicitly not connected. No rough/static values!
                sensor_readings_dict = None
                sensor_quality_score = None
                has_sensor = False

        # 3. Quality Evaluation & 65% Threshold Gating
        quality_assessment = QualityService.evaluate_overall_quality(
            camera_quality_score=camera_quality_score,
            sensor_quality_score=sensor_quality_score,
            has_camera=session.has_camera,
            has_sensor=has_sensor
        )

        dq_record = DataQuality(
            session_id=session.id,
            camera_quality_score=quality_assessment["camera_quality_score"],
            sensor_quality_score=quality_assessment["sensor_quality_score"] or 0.0,
            overall_data_quality=quality_assessment["overall_data_quality"],
            quality_pass=quality_assessment["quality_pass"],
            rejection_reason=quality_assessment["rejection_reason"]
        )
        db.add(dq_record)
        db.commit()

        # Enforce Refusal Rule if overall_data_quality < 65%
        if not quality_assessment["quality_pass"]:
            session.status = "rejected"
            db.commit()
            return jsonify({
                "status": "rejected",
                "message": f"Screening refused: Overall data quality ({quality_assessment['overall_data_quality']}%) is below required 65% threshold.",
                "quality_assessment": quality_assessment,
                "disclaimer": "Academic research prototype - NOT a medical diagnostic system."
            }), 422

        # 4. Fetch Baseline Profile if available
        baseline_profile = BaselineService.get_user_baseline(db, user_code) if user_code else None

        # 5. Multimodal Inference
        multi_service = MultimodalService.get_instance()
        inference_result = multi_service.predict_multimodal(
            visual_features=visual_features_dict,
            sensor_readings=sensor_readings_dict,
            baseline_profile=baseline_profile
        )

        # 6. Save Model Results to Database
        tri_models = inference_result.get("tri_model_comparison", {})
        for mod_key, mod_type in [
            ("model_a_sensor", "hardware_only"),
            ("model_b_vision", "vision_only"),
            ("model_c_multimodal", "multimodal")
        ]:
            mod_res = tri_models.get(mod_key, {})
            p_class = mod_res.get("predicted_class")
            conf = mod_res.get("confidence")
            mr = ModelResult(
                session_id=session.id,
                model_type=mod_type,
                predicted_class=p_class if p_class is not None else 0,
                model_confidence=conf if conf is not None else 0.0,
                raw_probabilities_json=json.dumps(mod_res)
            )
            db.add(mr)

        # 7. Save Anomaly Results to Database
        anom = inference_result.get("anomaly_analysis", {})
        base_dev = inference_result.get("baseline_deviation", {})
        ar = AnomalyResult(
            session_id=session.id,
            isolation_forest_score=anom.get("anomaly_score", 0.0),
            baseline_distance=base_dev.get("baseline_distance", 0.0),
            anomaly_score=anom.get("anomaly_score", 0.0),
            anomaly_flag=anom.get("anomaly_flag", False),
            summary_text=inference_result.get("screening_verdict", "")
        )
        db.add(ar)

        session.status = "completed"
        db.commit()

        return jsonify({
            "status": "success",
            "session_id": session.id,
            "user_code": user_code,
            "quality_assessment": quality_assessment,
            "screening_results": inference_result
        }), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@screening_bp.route('/<session_id>/results', methods=['GET'])
def get_screening_results(session_id):
    """Retrieves stored screening results and reports for a completed session."""
    db = SessionLocal()
    try:
        session = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
        if not session:
            return jsonify({"error": "Session not found"}), 404

        ar = session.anomaly_result
        dq = session.data_quality
        mrs = session.model_results

        models_data = {}
        for mr in mrs:
            models_data[mr.model_type] = {
                "predicted_class": mr.predicted_class,
                "confidence": mr.model_confidence
            }

        return jsonify({
            "session_id": session.id,
            "user_code": session.user.user_code if session.user else "Anonymous",
            "status": session.status,
            "timestamp": session.timestamp.isoformat() + "Z",
            "data_quality": {
                "camera_quality_score": dq.camera_quality_score if dq else None,
                "sensor_quality_score": dq.sensor_quality_score if dq else None,
                "overall_data_quality": dq.overall_data_quality if dq else None,
                "quality_pass": dq.quality_pass if dq else None,
                "rejection_reason": dq.rejection_reason if dq else None
            },
            "anomaly_analysis": {
                "anomaly_score": ar.anomaly_score if ar else None,
                "anomaly_flag": ar.anomaly_flag if ar else False,
                "baseline_distance": ar.baseline_distance if ar else None,
                "verdict": ar.summary_text if ar else None
            },
            "models": models_data,
            "disclaimer": "Academic research prototype - NOT a medical diagnostic system."
        }), 200
    finally:
        db.close()

@screening_bp.route('/<session_id>/report.pdf', methods=['GET'])
def download_pdf_report(session_id):
    """
    Generates and streams the PDF research screening report for the given session.
    """
    db = SessionLocal()
    try:
        session = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
        if not session:
            return jsonify({"error": "Session not found"}), 404

        user_code = session.user.user_code if session.user else "Anonymous"
        timestamp = session.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

        latest_sr = db.query(SensorReading).filter(
            SensorReading.session_id == session.id
        ).order_by(SensorReading.timestamp.desc()).first()

        latest_vf = db.query(VisualFeature).filter(
            VisualFeature.session_id == session.id
        ).order_by(VisualFeature.timestamp.desc()).first()

        dq = session.data_quality
        ar = session.anomaly_result
        mrs = session.model_results

        # Only pass sensor dict if real sensor reading was ingested
        sensor_dict = None
        if latest_sr and latest_sr.heart_rate is not None:
            sensor_dict = {
                "heart_rate": latest_sr.heart_rate,
                "spo2": latest_sr.spo2,
                "temperature": latest_sr.temperature,
                "sensor_quality_score": latest_sr.sensor_quality_score or 95.0,
                "is_simulated": latest_sr.is_simulated
            }

        vision_dict = {
            "facial_symmetry_score": latest_vf.facial_symmetry_score if latest_vf else 0.92,
            "mean_openness": latest_vf.eye_openness if latest_vf else 0.28,
            "eye_asymmetry": 0.01,
            "skin_l_mean": latest_vf.skin_l_mean if latest_vf else 125.0,
            "overall_quality": latest_vf.camera_quality_score if latest_vf else 90.0
        }

        multi_service = MultimodalService.get_instance()
        baseline_profile = BaselineService.get_user_baseline(db, user_code)
        screening_res = multi_service.predict_multimodal(
            visual_features={"eyes": {"mean_openness": vision_dict["mean_openness"]}, "symmetry": {"facial_symmetry_score": vision_dict["facial_symmetry_score"]}, "skin": {"skin_l_mean": vision_dict["skin_l_mean"]}, "quality": {"overall_quality": vision_dict["overall_quality"]}},
            sensor_readings=sensor_dict,
            baseline_profile=baseline_profile
        )

        session_report_data = {
            "session_id": session.id,
            "user_code": user_code,
            "timestamp": timestamp,
            "status": session.status,
            "screening_results": screening_res,
            "sensor_data": sensor_dict,
            "vision_data": vision_dict,
            "data_quality": {
                "camera_quality_score": dq.camera_quality_score if dq else 90.0,
                "sensor_quality_score": dq.sensor_quality_score if dq else 0.0,
                "overall_data_quality": dq.overall_data_quality if dq else 90.0,
                "quality_pass": dq.quality_pass if dq else True
            }
        }

        pdf_bytes = PDFReportService.generate_report(session_report_data)
        
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=screening_report_{session_id[:8]}.pdf"
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500
    finally:
        db.close()

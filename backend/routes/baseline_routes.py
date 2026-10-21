from flask import Blueprint, request, jsonify
from backend.database.connection import SessionLocal
from backend.services.baseline_service import BaselineService

baseline_bp = Blueprint('baseline_bp', __name__, url_prefix='/api/baselines')

@baseline_bp.route('/<user_code>', methods=['GET'])
def get_baseline(user_code):
    """Retrieves personal baseline profile for a user."""
    db = SessionLocal()
    try:
        profile = BaselineService.get_user_baseline(db, user_code)
        if not profile:
            return jsonify({
                "user_code": user_code,
                "has_baseline": False,
                "message": "No baseline established yet for this user."
            }), 200

        return jsonify({
            "user_code": user_code,
            "has_baseline": True,
            "profile": profile
        }), 200
    finally:
        db.close()

@baseline_bp.route('/<user_code>/update', methods=['POST'])
def update_baseline(user_code):
    """
    Updates or calculates a user's personal baseline from historical sessions.
    Optionally accepts a custom profile dict in request body.
    """
    data = request.get_json() or {}
    custom_profile = data.get('profile')

    db = SessionLocal()
    try:
        profile = BaselineService.update_user_baseline(db, user_code, custom_profile=custom_profile)
        return jsonify({
            "status": "success",
            "user_code": user_code,
            "profile": profile
        }), 200
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

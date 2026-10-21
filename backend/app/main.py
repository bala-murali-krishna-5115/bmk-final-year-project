import os
from pathlib import Path
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from sqlalchemy import text

from backend.app.config import Config, BASE_DIR
from backend.database.init_db import init_db
from backend.database.connection import engine

# Import REST API blueprints
from backend.routes.session_routes import session_bp
from backend.routes.sensor_routes import sensor_bp
from backend.routes.vision_routes import vision_bp
from backend.routes.screening_routes import screening_bp
from backend.routes.baseline_routes import baseline_bp

def create_app(config_class=Config):
    """Flask application factory."""
    frontend_dir = BASE_DIR / "frontend"
    static_dir = frontend_dir / "static"

    app = Flask(
        __name__,
        static_folder=str(static_dir),
        static_url_path='/static'
    )
    app.config.from_object(config_class)
    
    # Enable CORS for cross-origin frontend clients
    CORS(app)
    
    # Initialize database tables
    with app.app_context():
        init_db()

    # Register blueprints
    app.register_blueprint(session_bp)
    app.register_blueprint(sensor_bp)
    app.register_blueprint(vision_bp)
    app.register_blueprint(screening_bp)
    app.register_blueprint(baseline_bp)

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """System health check endpoint."""
        db_status = "connected"
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            db_status = f"error: {str(e)}"

        return jsonify({
            "status": "online",
            "service": "AI-Based Multimodal Health Screening Backend",
            "version": "1.0.0",
            "database": db_status,
            "disclaimer": "Academic research prototype - NOT a medical diagnostic system."
        }), 200

    @app.route('/', methods=['GET'])
    @app.route('/dashboard', methods=['GET'])
    def serve_dashboard():
        """Serve the interactive web screening dashboard."""
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return send_file(index_file)
        return jsonify({"message": "Frontend index.html not found", "api_health": "/api/health"}), 404

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.PORT, debug=True)

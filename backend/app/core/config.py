import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Based Multimodal Health Screening System"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Environment & Server Settings
    ENVIRONMENT: str = os.getenv("FLASK_ENV", "development")
    DEBUG: bool = True
    PORT: int = int(os.getenv("PORT", 8000))
    
    # CORS Configuration
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5000", "http://localhost:8000", "*"]

    # Database Settings
    DB_PATH: Path = BASE_DIR / "data" / "health_screening.db"
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

    # Model & Artifact Paths
    SCALER_PATH: str = "artifacts/scalers/visual_scaler.joblib"
    MODEL_PATH: str = "models/visual/visual_classifier.joblib"
    ANOMALY_PATH: str = "models/anomaly/visual_anomaly_detector.joblib"

    # Quality Thresholds
    MIN_CAMERA_QUALITY_SCORE: float = 65.0
    MIN_SENSOR_QUALITY_SCORE: float = 70.0

settings = Settings()

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

class Config:
    """System configuration parameters."""
    APP_ENV = os.getenv("APP_ENV", "development").lower()
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_12345")
    BASIC_AUTH_USERNAME = os.getenv("BASIC_AUTH_USERNAME")
    BASIC_AUTH_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD")
    PORT = int(os.getenv("PORT", 5000))
    
    # SQLite Database URI
    DB_PATH = BASE_DIR / "data" / "health_screening.db"
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Quality Engine Thresholds
    MIN_CAMERA_QUALITY_SCORE = float(os.getenv("MIN_CAMERA_QUALITY_SCORE", 65.0))
    MIN_SENSOR_QUALITY_SCORE = float(os.getenv("MIN_SENSOR_QUALITY_SCORE", 70.0))
    MIN_OVERALL_QUALITY_SCORE = float(os.getenv("MIN_OVERALL_QUALITY_SCORE", 65.0))

    # Physiological Hardware Sensor Ranges (Validation Rules)
    HR_MIN = 40.0
    HR_MAX = 200.0
    SPO2_MIN = 70.0
    SPO2_MAX = 100.0
    TEMP_MIN = 30.0
    TEMP_MAX = 43.0

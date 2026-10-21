from datetime import datetime
import uuid
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    """Anonymous user profile table."""
    __tablename__ = 'users'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_code = Column(String(50), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sessions = relationship("ScreeningSession", back_populates="user", cascade="all, delete-orphan")
    baseline = relationship("BaselineProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")

class BaselineProfile(Base):
    """Learned user baseline features for anomaly detection."""
    __tablename__ = 'baseline_profiles'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    baseline_vector_json = Column(Text, nullable=False)  # Stored JSON string
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="baseline")

class ScreeningSession(Base):
    """Screening session metadata."""
    __tablename__ = 'screening_sessions'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(20), default="created")  # created, in_progress, completed, rejected
    has_camera = Column(Boolean, default=True)
    has_sensors = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="sessions")
    sensor_readings = relationship("SensorReading", back_populates="session", cascade="all, delete-orphan")
    visual_features = relationship("VisualFeature", back_populates="session", cascade="all, delete-orphan")
    data_quality = relationship("DataQuality", back_populates="session", uselist=False, cascade="all, delete-orphan")
    model_results = relationship("ModelResult", back_populates="session", cascade="all, delete-orphan")
    anomaly_result = relationship("AnomalyResult", back_populates="session", uselist=False, cascade="all, delete-orphan")

class SensorReading(Base):
    """Physiological sensor measurement table."""
    __tablename__ = 'sensor_readings'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey('screening_sessions.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    heart_rate = Column(Float, nullable=True)
    spo2 = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    sensor_quality_score = Column(Float, default=100.0)
    is_simulated = Column(Boolean, default=False)

    # Relationship
    session = relationship("ScreeningSession", back_populates="sensor_readings")

class VisualFeature(Base):
    """Computer-vision derived facial and skin features."""
    __tablename__ = 'visual_features'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey('screening_sessions.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Eye Analysis
    ear_left = Column(Float, nullable=True)
    ear_right = Column(Float, nullable=True)
    eye_openness = Column(Float, nullable=True)
    
    # Symmetry Analysis
    facial_symmetry_score = Column(Float, nullable=True)
    
    # Skin Statistics (RGB, HSV, Lab, Texture)
    skin_r_mean = Column(Float, nullable=True)
    skin_g_mean = Column(Float, nullable=True)
    skin_b_mean = Column(Float, nullable=True)
    skin_h_mean = Column(Float, nullable=True)
    skin_s_mean = Column(Float, nullable=True)
    skin_v_mean = Column(Float, nullable=True)
    skin_l_mean = Column(Float, nullable=True)
    skin_a_mean = Column(Float, nullable=True)
    skin_b_lab_mean = Column(Float, nullable=True)
    skin_texture_lbp = Column(Float, nullable=True)
    
    # Quality Score
    camera_quality_score = Column(Float, default=100.0)

    # Relationship
    session = relationship("ScreeningSession", back_populates="visual_features")

class DataQuality(Base):
    """Quality engine assessment results."""
    __tablename__ = 'data_quality'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey('screening_sessions.id'), nullable=False)
    camera_quality_score = Column(Float, nullable=False)
    sensor_quality_score = Column(Float, nullable=False)
    overall_data_quality = Column(Float, nullable=False)
    quality_pass = Column(Boolean, default=True)
    rejection_reason = Column(Text, nullable=True)

    # Relationship
    session = relationship("ScreeningSession", back_populates="data_quality")

class ModelResult(Base):
    """Machine learning model outputs (Hardware-only, Vision-only, Multimodal)."""
    __tablename__ = 'model_results'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey('screening_sessions.id'), nullable=False)
    model_type = Column(String(30), nullable=False)  # 'hardware_only', 'vision_only', 'multimodal'
    predicted_class = Column(Integer, nullable=False)
    model_confidence = Column(Float, nullable=False)
    raw_probabilities_json = Column(Text, nullable=True)

    # Relationship
    session = relationship("ScreeningSession", back_populates="model_results")

class AnomalyResult(Base):
    """Personal baseline anomaly & pattern deviation outputs."""
    __tablename__ = 'anomaly_results'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey('screening_sessions.id'), nullable=False)
    isolation_forest_score = Column(Float, nullable=True)
    baseline_distance = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=False)  # Normalized 0.0 to 1.0
    anomaly_flag = Column(Boolean, default=False)
    summary_text = Column(Text, nullable=True)

    # Relationship
    session = relationship("ScreeningSession", back_populates="anomaly_result")

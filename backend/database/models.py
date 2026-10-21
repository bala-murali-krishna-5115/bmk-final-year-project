import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON, Index
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class Person(Base):
    """Anonymous person/user record."""
    __tablename__ = 'persons'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    external_reference = Column(String(50), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    sessions = relationship("ScreeningSession", back_populates="person", cascade="all, delete-orphan")

class ScreeningSession(Base):
    """Screening session record."""
    __tablename__ = 'screening_sessions'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    person_id = Column(String(36), ForeignKey('persons.id'), nullable=False, index=True)
    session_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(20), default="created")  # created, in_progress, completed, rejected
    duration = Column(Float, default=0.0)  # session duration in seconds
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    person = relationship("Person", back_populates="sessions")
    vision_features = relationship("VisionFeatures", back_populates="session", cascade="all, delete-orphan")
    sensor_measurements = relationship("SensorMeasurement", back_populates="session", cascade="all, delete-orphan")
    data_quality = relationship("DataQuality", back_populates="session", uselist=False, cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="session", cascade="all, delete-orphan")

class VisionFeatures(Base):
    """Stored visual features payload."""
    __tablename__ = 'vision_features'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey('screening_sessions.id'), nullable=False, index=True)
    feature_schema_version = Column(String(10), default="1.0")
    features_json = Column(JSON, nullable=False)  # Stores complete feature dictionary
    camera_quality_score = Column(Float, default=100.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("ScreeningSession", back_populates="vision_features")

class SensorMeasurement(Base):
    """Physiological sensor measurement table (prepared for Phase 5)."""
    __tablename__ = 'sensor_measurements'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey('screening_sessions.id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    heart_rate = Column(Float, nullable=True)
    spo2 = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    respiratory_rate = Column(Float, nullable=True)
    gsr = Column(Float, nullable=True)
    sensor_quality_score = Column(Float, default=100.0)
    is_simulated = Column(Boolean, default=False)

    # Relationship
    session = relationship("ScreeningSession", back_populates="sensor_measurements")

class Prediction(Base):
    """Model inference results."""
    __tablename__ = 'predictions'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey('screening_sessions.id'), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(10), default="1.0")
    prediction = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    prediction_status = Column(String(30), default="success")  # success, low_quality, insufficient_data
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("ScreeningSession", back_populates="predictions")

class DataQuality(Base):
    """Data quality layer record."""
    __tablename__ = 'data_quality'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey('screening_sessions.id'), nullable=False, index=True)
    camera_quality = Column(Float, nullable=False)
    sensor_quality = Column(Float, nullable=True)
    missing_features = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)
    quality_status = Column(String(20), default="good")  # good, poor, rejected

    # Relationship
    session = relationship("ScreeningSession", back_populates="data_quality")

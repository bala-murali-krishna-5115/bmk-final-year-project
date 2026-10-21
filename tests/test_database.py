import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models.database_models import Base, User, ScreeningSession, SensorReading, VisualFeature, DataQuality, ModelResult, AnomalyResult, BaselineProfile

@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing database models."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_create_user_and_session(db_session):
    """Test user creation and screening session association."""
    user = User(user_code="U_TEST_001")
    db_session.add(user)
    db_session.commit()

    assert user.id is not None
    assert user.user_code == "U_TEST_001"

    session = ScreeningSession(user_id=user.id, status="created")
    db_session.add(session)
    db_session.commit()

    assert session.id is not None
    assert session.user_id == user.id
    assert len(user.sessions) == 1

def test_sensor_reading_and_visual_feature_storage(db_session):
    """Test storing sensor and visual features for a session."""
    user = User(user_code="U_TEST_002")
    db_session.add(user)
    db_session.commit()

    session = ScreeningSession(user_id=user.id)
    db_session.add(session)
    db_session.commit()

    reading = SensorReading(
        session_id=session.id,
        heart_rate=75.0,
        spo2=98.0,
        temperature=36.6,
        is_simulated=True
    )
    feature = VisualFeature(
        session_id=session.id,
        ear_left=0.28,
        ear_right=0.29,
        facial_symmetry_score=0.94,
        camera_quality_score=90.0
    )
    db_session.add_all([reading, feature])
    db_session.commit()

    assert len(session.sensor_readings) == 1
    assert session.sensor_readings[0].heart_rate == 75.0
    assert len(session.visual_features) == 1
    assert session.visual_features[0].facial_symmetry_score == 0.94

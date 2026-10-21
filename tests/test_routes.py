import pytest
import json
from backend.app.main import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test /api/health endpoint."""
    res = client.get('/api/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "online"
    assert data["database"] == "connected"

def test_session_creation_and_listing(client):
    """Test session creation and retrieval."""
    res = client.post('/api/sessions', json={"user_code": "TEST_USER_99"})
    assert res.status_code == 201
    data = res.get_json()
    assert "session_id" in data
    assert data["user_code"] == "TEST_USER_99"
    session_id = data["session_id"]

    # List sessions
    list_res = client.get('/api/sessions')
    assert list_res.status_code == 200
    sessions = list_res.get_json()["sessions"]
    assert any(s["session_id"] == session_id for s in sessions)

def test_sensor_ingestion_and_simulation(client):
    """Test sensor measurement ingestion and simulation."""
    sess_res = client.post('/api/sessions', json={"user_code": "TEST_USER_SENS"})
    session_id = sess_res.get_json()["session_id"]

    # 1. Ingest valid measurement
    meas_payload = {
        "session_id": session_id,
        "heart_rate": 74.0,
        "spo2": 98.0,
        "temperature": 36.65,
        "is_simulated": True
    }
    res = client.post('/api/sensors/measurement', json=meas_payload)
    assert res.status_code == 201
    data = res.get_json()
    assert data["heart_rate"] == 74.0
    assert data["sensor_quality_score"] >= 80.0

    # 2. Simulate sensor readings
    sim_res = client.post('/api/sensors/simulate', json={"session_id": session_id, "mode": "normal"})
    assert sim_res.status_code == 201
    assert sim_res.get_json()["is_simulated"] is True

    # 3. Retrieve latest
    lat_res = client.get(f'/api/sensors/{session_id}/latest')
    assert lat_res.status_code == 200
    assert lat_res.get_json()["session_id"] == session_id

def test_screening_run_and_pdf_generation(client):
    """Test full multimodal screening run and PDF report generation."""
    sess_res = client.post('/api/sessions', json={"user_code": "TEST_USER_SCREEN"})
    session_id = sess_res.get_json()["session_id"]

    screening_payload = {
        "session_id": session_id,
        "user_code": "TEST_USER_SCREEN",
        "sensor_data": {
            "heart_rate": 72.0,
            "spo2": 98.5,
            "temperature": 36.6,
            "is_simulated": True
        }
    }
    res = client.post('/api/screening/run', json=screening_payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "screening_results" in data
    assert "tri_model_comparison" in data["screening_results"]
    assert "shap_attributions" in data["screening_results"]
    assert data["quality_assessment"]["quality_pass"] is True

    # Download PDF report
    pdf_res = client.get(f'/api/screening/{session_id}/report.pdf')
    assert pdf_res.status_code == 200
    assert pdf_res.mimetype == "application/pdf"
    assert len(pdf_res.data) > 1000  # Non-empty PDF document

def test_screening_refusal_on_low_quality(client):
    """Test mandatory 65% quality threshold refusal rule."""
    sess_res = client.post('/api/sessions', json={"user_code": "TEST_USER_LOWQ"})
    session_id = sess_res.get_json()["session_id"]

    # Provide sensor data with extreme out-of-bounds metrics (e.g. detached sensor)
    low_q_payload = {
        "session_id": session_id,
        "user_code": "TEST_USER_LOWQ",
        "sensor_data": {
            "heart_rate": 25.0,  # Below 40 bpm
            "spo2": 55.0,        # Below 70%
            "temperature": 25.0, # Below 30 C
            "is_simulated": True
        }
    }
    res = client.post('/api/screening/run', json=low_q_payload)
    assert res.status_code == 422
    data = res.get_json()
    assert data["status"] == "rejected"
    assert "quality_assessment" in data
    assert data["quality_assessment"]["quality_pass"] is False
    assert data["quality_assessment"]["overall_data_quality"] < 65.0

def test_baseline_routes(client):
    """Test personal baseline retrieval and calculation."""
    res = client.get('/api/baselines/TEST_USER_BASE')
    assert res.status_code == 200

    update_res = client.post('/api/baselines/TEST_USER_BASE/update', json={})
    assert update_res.status_code == 200
    assert "profile" in update_res.get_json()

def test_dashboard_route(client):
    """Test dashboard serving at root route."""
    res = client.get('/')
    assert res.status_code == 200
    assert b"AI-Based Multimodal Health Screening System" in res.data

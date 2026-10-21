import pytest
from backend.app.main import create_app

@pytest.fixture
def client():
    """Create Flask test client."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check_endpoint(client):
    """Test /api/health endpoint returns HTTP 200 and online status."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'online'
    assert data['database'] == 'connected'
    assert 'disclaimer' in data

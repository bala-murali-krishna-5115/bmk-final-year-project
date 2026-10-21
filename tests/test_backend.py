import pytest
from backend.app.main import create_app

class ProductionConfig:
    APP_ENV = "production"
    BASIC_AUTH_USERNAME = "demo"
    BASIC_AUTH_PASSWORD = "change-this-password"

@pytest.fixture
def client():
    """Create Flask test client."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_production_routes_require_basic_auth():
    app = create_app(ProductionConfig)
    client = app.test_client()

    assert client.get("/").status_code == 401
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/health").status_code == 200
    assert client.get("/", auth=("demo", "change-this-password")).status_code == 200

def test_production_requires_auth_configuration():
    class IncompleteProductionConfig:
        APP_ENV = "production"
        BASIC_AUTH_USERNAME = "demo"
        BASIC_AUTH_PASSWORD = None

    with pytest.raises(RuntimeError, match="BASIC_AUTH_USERNAME and BASIC_AUTH_PASSWORD"):
        create_app(IncompleteProductionConfig)

def test_health_check_endpoint(client):
    """Test /api/health endpoint returns HTTP 200 and online status."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'online'
    assert data['database'] == 'connected'
    assert 'disclaimer' in data

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from backend.app.config import Config
from backend.models.database_models import Base

# Ensure the database directory exists
Config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create engine
engine = create_engine(
    Config.SQLALCHEMY_DATABASE_URI,
    connect_args={"check_same_thread": False} if "sqlite" in Config.SQLALCHEMY_DATABASE_URI else {}
)

# Create session factory
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def get_db():
    """Dependency / Helper function to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

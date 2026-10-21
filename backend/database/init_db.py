import logging
from backend.database.connection import engine
from backend.models.database_models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Create all database tables defined in database_models.py."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully.")

if __name__ == "__main__":
    init_db()

"""Detectify Database Session Factory."""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from detectify.config import settings
from detectify.db.models import Base


def get_engine():
    """Create database engine based on DATABASE_URL."""
    connect_args = {}
    
    # SQLite-specific settings
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    
    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        echo=settings.log_level == "DEBUG",
    )
    return engine


# Create engine and session factory
engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all tables in the database."""
    Base.metadata.drop_all(bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI - yields database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

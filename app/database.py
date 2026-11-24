"""Database connection, engine, session factory, and initialization using SQLAlchemy."""

import logging
from typing import Generator
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# For SQLite, check_same_thread=False is needed for multi-threaded FastAPI workers
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

# Enable foreign key constraint enforcement and WAL mode for concurrent performance in SQLite
if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db() -> None:
    """Initializes the database by creating declared tables and running idempotent migrations."""
    try:
        # Import models here to ensure they are registered with Base.metadata
        import app.models  # noqa: F401
        Base.metadata.create_all(bind=engine)

        # Idempotent migration for existing SQLite databases
        with engine.begin() as conn:
            inspector = inspect(conn)
            if "users" in inspector.get_table_names():
                existing_user_cols = [c["name"] for c in inspector.get_columns("users")]
                new_user_cols = [
                    ("auth_provider", "VARCHAR(50) DEFAULT 'local'"),
                    ("google_id", "VARCHAR(255)"),
                    ("avatar_url", "VARCHAR(1024)"),
                    ("role", "VARCHAR(50) DEFAULT 'student'"),
                ]
                for col_name, col_type in new_user_cols:
                    if col_name not in existing_user_cols:
                        conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
                        logger.info(f"Migrated users table: added column {col_name}.")

            if "answers" in inspector.get_table_names():
                existing_cols = [c["name"] for c in inspector.get_columns("answers")]
                new_answer_cols = [
                    ("session_id", "INTEGER REFERENCES interview_sessions(id) ON DELETE CASCADE"),
                    ("correctness_status", "VARCHAR(50)"),
                    ("evaluation_feedback", "TEXT"),
                    ("correct_concepts", "TEXT"),
                    ("incorrect_concepts", "TEXT"),
                    ("missing_concepts", "TEXT"),
                    ("correct_answer", "TEXT"),
                    ("simple_explanation", "TEXT"),
                    ("detailed_explanation", "TEXT"),
                    ("example", "TEXT"),
                    ("key_points_to_remember", "TEXT"),
                    ("improved_answer", "TEXT"),
                    ("evaluation_details", "TEXT"),
                    ("model_answer_provider", "VARCHAR(50) DEFAULT 'Google Gemini'"),
                    ("model_answer_error", "TEXT"),
                    ("model_answer_disclaimer", "TEXT"),
                ]
                for col_name, col_type in new_answer_cols:
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE answers ADD COLUMN {col_name} {col_type}"))
                        logger.info(f"Migrated answers table: added column {col_name}.")

                # Backfill session_id on answers from questions table if currently null
                try:
                    conn.execute(text(
                        "UPDATE answers SET session_id = (SELECT session_id FROM questions WHERE questions.id = answers.question_id) WHERE session_id IS NULL"
                    ))
                except Exception as backfill_exc:
                    logger.debug(f"Answers session_id backfill skipped: {backfill_exc}")

            if "session_reports" in inspector.get_table_names():
                existing_rep_cols = [c["name"] for c in inspector.get_columns("session_reports")]
                new_cols = [
                    ("questions_answered", "INTEGER DEFAULT 0"),
                    ("average_score", "REAL"),
                    ("strongest_topics", "TEXT"),
                    ("weakest_topics", "TEXT"),
                    ("technical_assessment", "TEXT"),
                    ("communication_feedback", "TEXT"),
                    ("repeated_mistakes", "TEXT"),
                    ("suggested_next_topics", "TEXT"),
                    ("report_details", "TEXT"),
                ]
                for col_name, col_type in new_cols:
                    if col_name not in existing_rep_cols:
                        conn.execute(text(f"ALTER TABLE session_reports ADD COLUMN {col_name} {col_type}"))
                        logger.info(f"Migrated session_reports table: added column {col_name}.")

        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        logger.error(f"Error during database initialization: {exc}")
        raise


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a transactional database session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

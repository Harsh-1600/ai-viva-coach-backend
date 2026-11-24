"""Application configuration settings loaded from environment variables."""

import os
from functools import lru_cache
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ENV_FILE = os.path.join(_BACKEND_DIR, ".env")
_DEFAULT_DB_PATH = os.path.join(_BACKEND_DIR, "viva_coach.db")
_DEFAULT_DB_URL = f"sqlite:///{_DEFAULT_DB_PATH.replace(os.sep, '/')}"


class Settings(BaseSettings):
    """Configuration settings for the AI Viva & Interview Preparation Coach."""

    # Project Information
    PROJECT_NAME: str = "AI Viva & Interview Preparation Coach"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"

    # AI Configuration (Anthropic Claude)
    # Default to "mock" for offline testing; overridden by real key in .env or environment
    ANTHROPIC_API_KEY: str = "mock"
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"

    # AI Configuration (OpenAI GPT)
    # Default to "mock" for offline testing; overridden by real key in .env or environment
    OPENAI_API_KEY: str = "mock"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # AI Configuration (Google Gemini - legacy fallback)
    # Default to "mock" for offline testing; overridden by real key in .env or environment
    GEMINI_API_KEY: str = "mock"
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # AI Provider Delegation Design
    EVALUATION_PROVIDER: str = "claude"
    MODEL_ANSWER_PROVIDER: str = "openai"

    # Security & Authentication
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production-viva-coach"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    TEACHER_INVITATION_CODE: str = "TEACH-COACH-2026"

    # SMTP & Automatic Task Reminder Email Configuration
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: str = "noreply@vivacoach.edu"
    EMAIL_REMINDERS_ENABLED: bool = True
    EMAIL_REMINDER_COOLDOWN_HOURS: int = 24

    # Google OAuth Configuration
    GOOGLE_CLIENT_ID: Optional[str] = None

    # Database Configuration
    DATABASE_URL: str = _DEFAULT_DB_URL

    # Server Configuration
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    PORT: Optional[int] = None  # Dynamic port provided by cloud environments (Render, Railway, etc.)
    FRONTEND_URL: str = "http://localhost:5173"

    @property
    def effective_port(self) -> int:
        """Returns the dynamic hosting platform port ($PORT) if set, otherwise BACKEND_PORT."""
        return self.PORT if self.PORT is not None else self.BACKEND_PORT

    # Allowed CORS Origins (accepts JSON list or comma-separated string)
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost",
        "https://localhost",
        "capacitor://localhost",
        "ionic://localhost",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://172.1.10.41:5173",
        "http://172.1.10.41:8000",
        "http://172.3.1.209:8000",
        "http://172.3.1.21:5173",
        "http://172.3.1.21:8000",
        "http://10.14.195.181:5173",
        "http://10.14.195.181:8000",
        "http://10.14.195.181",
        "https://ai-viva-coach.web.app",
        "https://ai-viva-coach.firebaseapp.com",
    ]

    @field_validator("CORS_ORIGINS", mode="after")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Accepts either JSON array list or comma-separated string from cloud env vars."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=(_DEFAULT_ENV_FILE, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns cached application settings instance."""
    return Settings()

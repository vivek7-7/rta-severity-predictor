"""
app/config.py
Application configuration using pydantic-settings.
All secrets must be provided via environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for the RTA Severity Predictor application."""

    # App
    APP_NAME: str = "RTA Severity Predictor"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = Field(
        default="change-this-in-production-use-a-long-random-string-32chars+",
        description="JWT signing secret — override via SECRET_KEY env var",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/rta.db"

    # ML artifacts directory
    ARTIFACTS_DIR: Path = BASE_DIR / "app" / "ml" / "artifacts"

    # Default model to use for predictions
    DEFAULT_MODEL: str = "xgb"

    # Pagination
    HISTORY_PAGE_SIZE: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    
    class Settings(BaseSettings):
    APP_NAME: str = "CrashSense"
    APP_VERSION: str = "2.0.0"
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/rta.db"
    HISTORY_PAGE_SIZE: int = 20        # ← add this line

settings = Settings()

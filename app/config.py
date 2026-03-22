from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "CrashSense"
    APP_VERSION: str = "2.0.0"
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/rta.db"
    HISTORY_PAGE_SIZE: int = 20

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## File 2: `app/templates/predict.html` — only line 30

Find:
```
{{ info.name }} — {{ info.type }}{% if info.default %} ★{% endif %}
```
Replace with:
```
{{ info.name }}{% if info.default %} ★{% endif %}

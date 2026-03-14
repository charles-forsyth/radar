from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DB_URL: str = "sqlite+aiosqlite:///radar.db"

    # Cloud SQL IAM Auth Settings (Deprecated in local-only mode)
    INSTANCE_CONNECTION_NAME: str | None = None
    ENABLE_IAM_AUTH: bool = False
    DB_USER: str = "postgres"
    DB_PASSWORD: str | None = None
    DB_NAME: str = "radar"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

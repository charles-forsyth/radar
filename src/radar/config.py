from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DB_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/radar"
    GEMINI_API_KEY: str | None = None

    # Cloud SQL IAM Auth Settings
    INSTANCE_CONNECTION_NAME: str | None = None
    ENABLE_IAM_AUTH: bool = False
    DB_USER: str = "postgres"  # For IAM, this would be the SA email or IAM user
    DB_PASSWORD: str | None = None
    DB_NAME: str = "radar"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DB_URL: str = "sqlite+aiosqlite:///radar.db"

    # Paths to local binaries and scripts
    ATMOS_BIN: str = "/home/chuck/.local/bin/atmos"
    ROAM_BIN: str = "/home/chuck/.local/bin/roam"
    PYTHON_BIN: str = "/home/chuck/bin/python3"
    VOICE_SCRIPT: str = "/home/chuck/Scripts/generate_voice.py"

    # Internal tool paths
    TOOL_EMBED: str = "src/radar/tools/radar_embed"
    TOOL_EXTRACT: str = "src/radar/tools/radar_extract"
    TOOL_SUMMARIZE: str = "src/radar/tools/radar_summarize"
    TOOL_FETCH: str = "src/radar/tools/radar_fetch"

    # Tactical settings
    HOME_COORDS: tuple[float, float] = (41.9168, -77.1042)
    SECTOR_RADIUS_MILES: int = 150

    # Model settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

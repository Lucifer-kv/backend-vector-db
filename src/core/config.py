from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    COHERE_API_KEY: str
    BASE_URL: str = "http://localhost:8000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
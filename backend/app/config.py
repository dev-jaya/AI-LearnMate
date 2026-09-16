from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server-side application configuration.

    GEMINI_API_KEY is intentionally backend-only and must never be exposed to
    the browser or returned by an API response.
    """

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    database_url: str = "sqlite:///./ai_learnmate.db"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

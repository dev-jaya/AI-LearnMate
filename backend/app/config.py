from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server-side application configuration.

    GEMINI_API_KEY is intentionally backend-only and must never be exposed to
    the browser or returned by an API response.
    """

    gemini_api_key: str = ""
    # Gemini 3.8 Flash is the current primary model for the assistant.
    # Render can still override this value through GEMINI_MODEL.
    gemini_model: str = "gemini-3.8-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    database_url: str = "sqlite:///./ai_learnmate.db"
    frontend_origins: str = (
        "http://localhost:5173,"
        "http://localhost:5500,"
        "http://127.0.0.1:5173,"
        "http://127.0.0.1:5500,"
        "https://ai-learnmate-frontend.onrender.com"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()

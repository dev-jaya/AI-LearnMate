from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ai_provider: str = "fallback"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    hf_base_url: str = ""
    hf_api_key: str = ""
    hf_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    database_url: str = "sqlite:///./ai_learnmate.db"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

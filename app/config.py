from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/what_the_docs"
    )
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    embedding_dim: int = 1536
    upload_dir: str = "/tmp/what_the_docs_uploads"


settings = Settings()

EMBEDDING_DIM = settings.embedding_dim
DATABASE_URL = settings.database_url
REDIS_URL = settings.redis_url

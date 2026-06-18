from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    # LLM config
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # App config
    APP_NAME: str = "AI Scientific Intelligence Platform"
    ENV: str = os.getenv("ENV", "dev")

    # Vector DB config
    CHROMA_PATH: str = "data/chroma"

    # Embedding model
    EMBEDDING_MODEL: str = (
    os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2"
    )
)


settings = Settings()
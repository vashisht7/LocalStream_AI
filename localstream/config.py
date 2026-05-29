import os
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Set up standard logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("localstream.config")

class Settings(BaseSettings):
    # Storage paths
    storage_dir: str = Field(default="storage/uploads")
    qdrant_data_dir: str = Field(default="storage/qdrant_data")
    
    # Vector DB settings
    collection_name: str = Field(default="multimedia_knowledge")
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    qdrant_url: str = Field(default="http://localhost:6333")
    
    # Whisper settings
    whisper_model_size: str = Field(default="base")

    # Read from environment variables if present
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def compute_device(self) -> str:
        # Check if USE_GPU=1 is set in environment
        use_gpu = os.environ.get("USE_GPU") == "1"
        if use_gpu:
            logger.info("USE_GPU=1 detected. Using CUDA compute device.")
            return "cuda"
        else:
            logger.info("USE_GPU=1 not set. Defaulting to CPU compute device.")
            return "cpu"

settings = Settings()

# Automatically create the required directories
try:
    os.makedirs(settings.storage_dir, exist_ok=True)
    logger.info(f"Verified/created upload directory: {settings.storage_dir}")
    os.makedirs(settings.qdrant_data_dir, exist_ok=True)
    logger.info(f"Verified/created Qdrant storage directory: {settings.qdrant_data_dir}")
except Exception as e:
    logger.error(f"Error creating directories: {e}")

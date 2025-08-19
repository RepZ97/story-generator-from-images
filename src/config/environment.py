import os
from dotenv import load_dotenv
from config.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable is required")
    raise ValueError("OPENAI_API_KEY environment variable is required")

logger.info("Environment configuration loaded successfully")

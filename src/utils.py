import os
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger(__name__)

def read_images_on_folder(folder_path):
    """Reads all image files in a specified folder and returns their paths."""

    image_paths = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff")

    if not os.path.isdir(folder_path):
        logger.error(f"The folder at '{folder_path}' does not exist.")
        return image_paths

    for filename in os.listdir(folder_path):
        # Check if the file has a valid image extension
        if filename.lower().endswith(valid_extensions):
            full_path = os.path.join(folder_path, filename)
            image_paths.append(full_path)

    logger.info(f"Found {len(image_paths)} image files in {folder_path}")
    return image_paths


def get_file_timestamp(file_path):
    """Get timestamp from file metadata or use current time as fallback."""
    try:
        # Get the file modification time
        stat = os.stat(file_path)
        timestamp = datetime.fromtimestamp(stat.st_mtime)
    except Exception as e:
        # Fallback to current time
        logger.warning(f"Could not get file timestamp for {file_path}: {e}. Using current time.")
        timestamp = datetime.now()

    return timestamp.isoformat() + "Z"


def clean_json_text(text: str) -> str:
    """Clean JSON text by removing markdown code blocks."""
    content = text.strip()
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    elif content.startswith("```"):
        content = content[3:]  # Remove ```
    if content.endswith("```"):
        content = content[:-3]  # Remove ```
    return content.strip()

import logging
from datetime import datetime
import os
import re

def setup_logger(name: str = "kc_agent", log_level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with both console and file handlers.

    Args:
        name: Name of the logger
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent adding handlers multiple times in case of multiple calls
    if logger.hasHandlers():
        return logger

    # Create formatters
    log_format = "[%(name)s] - %(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "logger"
        log_file = os.path.join(log_dir, f"{safe_name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass

    return logger

if __name__ == "__main__":
    logger = setup_logger(name="test")
    logger.info("Logger initialized")
# Set up the logger at module level
# logger = setup_logger()

# Example usage in your class:
# logger.debug("Debug message")
# logger.info("Info message")
# logger.warning("Warning message")
# logger.error("Error message")
# logger.critical("Critical message")

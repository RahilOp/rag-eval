import logging
import os

def configure_logger(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)  # Ensure the logs directory exists
    log_file = os.path.abspath(os.path.join(log_dir, "final_pipeline_logs.log"))
    logger = logging.getLogger("pipeline_logger")

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        logger.setLevel(logging.WARNING)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

        print(f"Handlers: {[type(handler).__name__ for handler in logger.handlers]}")
        print(f"Log file: {log_file}")

    return logger

logger = configure_logger()

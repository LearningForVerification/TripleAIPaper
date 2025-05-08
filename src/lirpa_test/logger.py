# logger_setup.py
import logging
from logging import StreamHandler, FileHandler
import os


class ColorFormatter(logging.Formatter):
    COLOR_RESET = "\033[0m"
    COLOR_MAP = {
        'regularized_trainer.py': '\033[94m',  # blu
        'hyper_param_search.py': '\033[92m',  # verde
    }

    def format(self, record):
        filename = record.pathname.split(os.sep)[-1]  # solo nome file
        color = self.COLOR_MAP.get(filename, "")
        formatted = super().format(record)
        return f"{color}{formatted}{self.COLOR_RESET}"


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Changed to DEBUG level

    # Properly close existing handlers before clearing
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            handler.close()  # Close the handler
            logger.removeHandler(handler)  # Remove the handler

    # Console handler with colors
    ch = StreamHandler()
    ch.setLevel(logging.DEBUG)  # Already at DEBUG level
    ch.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s'))

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # File handler without colors
    fh = FileHandler("logs/app.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

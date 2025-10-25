import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

def log_setup(logger_name=""):
    global logger
    # Generate a unique log file for each execution
    log_filename = datetime.now().strftime(logger_name+"log_daily_operation_%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    # Set up logging
    logger = logging.getLogger("ExecutionLogger")
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(log_filepath)

    # Set log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Example log entries

    # logger.warning("This is a warning message.")
    # logger.error("This is an error message.")

def log_info(msg):
    logger.info(msg)
def log_error(msg):
    logger.error(msg)
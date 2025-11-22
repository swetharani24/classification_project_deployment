import logging
import os

def setup_logging(script_name, folder_name='log'):
    """
    Sets up a logger that writes to both console and a log file.
    Automatically creates the folder if missing.
    """
    # Absolute path to the log folder
    log_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(log_dir, exist_ok=True)  # create folder if missing

    # Full path to the log file
    log_file_path = os.path.join(log_dir, f'{script_name}.log')

    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

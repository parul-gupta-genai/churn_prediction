import logging
import sys
from src.config import LOG_LEVEL, LOGS_DIR

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already configured
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(LOG_LEVEL)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    log_file = LOGS_DIR / "pipeline.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

logger = setup_logger("churn_prediction")

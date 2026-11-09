"""Route client logs to a run or diagnostic directory."""
import logging
from pathlib import Path
from common.paths import PROJECT_ROOT

def configure_client_log(directory=None):
    directory = Path(directory) if directory is not None else PROJECT_ROOT / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("projectairsim")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.FileHandler(directory / "projectairsim_client.log", mode="a"),
                    logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

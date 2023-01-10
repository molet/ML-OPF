from functools import lru_cache
import logging
import sys

@lru_cache()
def getLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
    stream_handler.setFormatter(formatter)
    return logger

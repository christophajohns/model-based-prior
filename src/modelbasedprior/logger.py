import logging

# Create and configure logger
def setup_logger(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s : %(name)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)
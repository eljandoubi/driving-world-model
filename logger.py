import logging

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(rank: int = 0):
    # ✅ Inject rank into ALL log records globally
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)

    logger = logging.getLogger()
    logger.handlers.clear()

    logger.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | Rank: %(rank)d | %(name)s | %(message)s"
    )

    if rank == 0:
        console = TqdmLoggingHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        logger.setLevel(logging.INFO)

    # 🔇 Silence noisy libs
    for name in ["httpx", "urllib3", "huggingface_hub"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    return logging.getLogger(__name__)
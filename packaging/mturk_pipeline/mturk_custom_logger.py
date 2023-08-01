import logging


def red(s):
    return "\033[91m{}\033[0m".format(s)


def green(s):
    return "\033[92m{}\033[0m".format(s)


def yellow(s):
    return "\033[93m{}\033[0m".format(s)


def white(s):
    return "\033[97m{}\033[0m".format(s)


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: white(format),
        logging.INFO: green(format),
        logging.WARNING: yellow(format),
        logging.ERROR: red(format),
        logging.CRITICAL: red(format)
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.handlers = []  # remove all existing handlers
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger

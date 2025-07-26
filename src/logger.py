import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(levelname)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "my_app": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}


_logging_configured = False


def get_logger(name: str) -> logging.Logger:
    global _logging_configured
    if not _logging_configured:
        logging.config.dictConfig(LOGGING_CONFIG)
        _logging_configured = True
        logging.getLogger(__name__).info("ロガー設定が完了しました。")

    return logging.getLogger(name)

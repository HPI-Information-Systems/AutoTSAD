import contextlib
import logging
import logging.handlers
import multiprocessing
from typing import Callable, Generator, Optional, Dict, Any, Union

from autotsad.config import GeneralSection, OptimizationSection

LOGGING_Q: multiprocessing.Queue = multiprocessing.Manager().Queue(-1)


@contextlib.contextmanager
def setup_logging_basic(
        stream=None,
        filename=None,
        filemode: str = "a",
        datefmt: Optional[str] = None,
        style: str = "%",
        fmt: str = logging.BASIC_FORMAT,
        level: Optional[Union[str, int]] = None,
        logger_configs: Optional[Dict[str, Any]] = None,
) -> Generator[None, None, None]:
    config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {},
        "handlers": {},
    }

    if stream is not None and filename is not None:
        raise ValueError("'stream' and 'filename' should not be specified together")

    # set up formatter
    config["formatters"]["default"] = {
        "class": "logging.Formatter",
        "format": fmt,
        "datefmt": datefmt,
        "style": style,
    }

    # set up handler
    if filename is not None:
        config["handlers"]["default"] = {
            "class": "logging.FileHandler",
            "filename": filename,
            "mode": filemode,
            "formatter": "default",
        }
    else:
        config["handlers"]["default"] = {
            "class": "logging.StreamHandler",
            "stream": stream,
            "formatter": "default",
        }

    # set up root logger
    config["root"] = {
        "handlers": ["default"],
    }
    if level is not None:
        if isinstance(level, int):
            level = logging.getLevelName(level)
        config["root"]["level"] = level

    if logger_configs is not None:
        config["loggers"] = logger_configs

    stop_logging = setup_logging(config)
    try:
        yield
    finally:
        stop_logging()


@contextlib.contextmanager
def setup_logging_from_config(general: GeneralSection, optimization: OptimizationSection) -> Generator[None, None, None]:
    filename = str(general.result_dir() / "autotsad.log")
    config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "default": {
                "class": "logging.Formatter",
                "format": "%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": filename,
                "mode": "a",
                "formatter": "default",
            }
        },
        "root": {
            "handlers": ["file"],
            "level": "NOTSET",
        },
        "loggers": {
            "AUTOTSAD": {"level": logging.getLevelName(general.logging_level)},
            "autotsad": {"level": logging.getLevelName(general.logging_level)},
            "numba": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
            "matplotlib": {"level": "WARNING"},
            "optuna": {"level": logging.getLevelName(optimization.optuna_logging_level)},
        }
    }
    stop_logging = setup_logging(config)
    try:
        yield
    finally:
        stop_logging()


def setup_logging(config: Dict[str, Any]) -> Callable[[], None]:
    stop_event = multiprocessing.Event()
    log_listener = multiprocessing.Process(
        target=logging_listener_process,
        name="AutoTSAD logging listener",
        args=(LOGGING_Q, stop_event, config)
    )
    log_listener.start()

    main_logging_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "handlers": {
            "queue": {
                "class": "logging.handlers.QueueHandler",
                "queue": LOGGING_Q
            }
        },
        "root": {
            "handlers": ["queue"],
            "level": "DEBUG"
        }
    }
    if "loggers" in config:
        main_logging_config["loggers"] = config["loggers"]
    logging.config.dictConfig(main_logging_config)
    # optuna.logging.set_verbosity()

    log = logging.getLogger("root")
    log.info("Logging started!")

    def stop() -> None:
        log.info("Stopping logging!")
        stop_event.set()
        log_listener.join()

    return stop


class MainQueueHandler(logging.Handler):
    """A simple handler for logging events received via the queue. Should only be used
    in its separate logging process.
    """
    def handle(self, record):
        if record.name == "root":
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(record.name)

        if logger.isEnabledFor(record.levelno):
            logger.handle(record)


def logging_listener_process(queue, stop_event, logging_config):
    logging.config.dictConfig(logging_config)
    listener = logging.handlers.QueueListener(queue, MainQueueHandler())
    listener.start()
    stop_event.wait()
    listener.stop()


@contextlib.contextmanager
def setup_process_logging(q: multiprocessing.Queue) -> Generator[None, None, None]:
    logging_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "handlers": {
            "queue": {
                "class": "logging.handlers.QueueHandler",
                "queue": q
            }
        },
        "root": {
            "handlers": ["queue"],
            "level": "DEBUG"
        }
    }
    logging.config.dictConfig(logging_config)
    yield multiprocessing.current_process().pid

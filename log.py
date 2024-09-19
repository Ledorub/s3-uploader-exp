import logging
import queue
from enum import Enum
from logging.handlers import QueueHandler
from queue import Queue
from threading import Thread
from typing import Protocol, cast

type LogQueue = Queue[logging.LogRecord | None]


class Level(int, Enum):
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0


class Logger(Protocol):
    level: Level

    def setLevel(self, level: Level) -> None: ...

    def critical(self, msg: str, *args, **kwargs) -> None: ...

    def fatal(self, msg: str, *args, **kwargs) -> None: ...

    def error(self, msg: str, *args, **kwargs) -> None: ...

    def warning(self, msg: str, *args, **kwargs) -> None: ...

    def warn(self, msg: str, *args, **kwargs) -> None: ...

    def info(self, msg: str, *args, **kwargs) -> None: ...

    def debug(self, msg: str, *args, **kwargs) -> None: ...

    def handle(self, record: logging.LogRecord, *args, **kwargs) -> None: ...


logger: Logger


def init_logger(
    log_queue: LogQueue, name: str, level: Level | int = Level.INFO
) -> None:
    if level // 10 < 0 or level // 10 > 5:
        raise ValueError(f"Invalid log level: {level}")

    l = logging.getLogger(name)
    l.addHandler(QueueHandler(log_queue))
    l.setLevel(level)
    global logger
    logger = cast(Logger, l)


def get_logger() -> Logger:
    return logger


class LogWriter:
    def __init__(self, log_queue: LogQueue, level: Level) -> None:
        self._level = level
        self._queue = log_queue
        self._logger = self._get_logger("LogWriter")
        self._thread = None

    def start(self):
        t = Thread(target=self._writer, args=(self._queue, self._logger))
        t.start()
        self._thread = t

    def stop(self):
        if self._thread is not None:
            self._thread.join()

    def _writer(self, log_queue: LogQueue, logger: Logger) -> None:
        while True:
            try:
                record = log_queue.get()
            except queue.Empty:
                continue

            if record is None:
                break
            logger.handle(record)

    def _get_logger(self, name: str) -> Logger:
        logger = logging.getLogger(name)

        handler = logging.StreamHandler()
        handler.setLevel(self._level)
        handler.setFormatter(logging.Formatter(fmt=logging.BASIC_FORMAT))
        logger.addHandler(handler)
        return cast(Logger, logger)

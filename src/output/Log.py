import logging
from typing import TYPE_CHECKING

import structlog

from output.run_artifacts import get_run_output_dir

if TYPE_CHECKING:
    from config.ConfigLoader import GlobalConfig


class _ContextLog:
    def __init__(self, context: dict):
        self._context = context

    def _emit(self, level: str, event: str, **kwargs):
        logger = structlog.get_logger().bind(**self._context)
        getattr(logger, level)(event, **kwargs)

    def information(self, event: str, **kwargs):
        self._emit("info", event, **kwargs)

    def debug(self, event: str, **kwargs):
        self._emit("debug", event, **kwargs)

    def warning(self, event: str, **kwargs):
        self._emit("warning", event, **kwargs)

    def error(self, event: str, **kwargs):
        self._emit("error", event, **kwargs)

    def critical(self, event: str, **kwargs):
        self._emit("critical", event, **kwargs)

    def exception(self, event: str, **kwargs):
        logger = structlog.get_logger().bind(**self._context)
        logger.error(event, exc_info=True, **kwargs)


class _Log:
    def initialize(self, config: "GlobalConfig"):
        log_dir = get_run_output_dir(config)
        log_file = log_dir / "log.txt"

        timestamper = structlog.processors.TimeStamper(fmt="iso")
        pre_chain = [timestamper, structlog.processors.add_log_level]

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(),
                foreign_pre_chain=pre_chain,
            )
        )

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=pre_chain,
            )
        )

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                timestamper,
                structlog.processors.add_log_level,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.information("logging_configured", log_file=str(log_file))

    def for_source(self, source: str) -> _ContextLog:
        return _ContextLog({"source": source})

    def for_context(self, **kwargs) -> _ContextLog:
        return _ContextLog(dict(kwargs))

    def information(self, event: str, **kwargs):
        structlog.get_logger().info(event, **kwargs)

    def debug(self, event: str, **kwargs):
        structlog.get_logger().debug(event, **kwargs)

    def warning(self, event: str, **kwargs):
        structlog.get_logger().warning(event, **kwargs)

    def error(self, event: str, **kwargs):
        structlog.get_logger().error(event, **kwargs)

    def critical(self, event: str, **kwargs):
        structlog.get_logger().critical(event, **kwargs)

    def exception(self, event: str, **kwargs):
        structlog.get_logger().error(event, exc_info=True, **kwargs)


Log = _Log()

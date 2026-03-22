import logging
import sys
from typing import TYPE_CHECKING, Any

from output.run_artifacts import get_run_output_dir

if TYPE_CHECKING:
    from config.ConfigLoader import GlobalConfig


class _ContextLog:
    def __init__(self, logger: logging.Logger, context: dict[str, Any] = None):
        self._logger = logger
        self._context = context or {}

    def _format_msg(self, event: str, **kwargs: Any) -> str:
        all_context = {**self._context, **kwargs}
        if not all_context:
            return event
        kv_pairs = [f"{k}={v}" for k, v in all_context.items()]
        return f"{event} | {', '.join(kv_pairs)}"

    def information(self, event: str, **kwargs: Any):
        self._logger.info(self._format_msg(event, **kwargs), extra={"device": "n/a"})

    def debug(self, event: str, **kwargs: Any):
        self._logger.debug(self._format_msg(event, **kwargs), extra={"device": "n/a"})

    def warning(self, event: str, **kwargs: Any):
        self._logger.warning(self._format_msg(event, **kwargs), extra={"device": "n/a"})

    def error(self, event: str, **kwargs: Any):
        self._logger.error(self._format_msg(event, **kwargs), extra={"device": "n/a"})

    def critical(self, event: str, **kwargs: Any):
        self._logger.critical(
            self._format_msg(event, **kwargs), extra={"device": "n/a"}
        )

    def exception(self, event: str, **kwargs: Any):
        self._logger.exception(
            self._format_msg(event, **kwargs), extra={"device": "n/a"}
        )


class _DeviceFilter(logging.Filter):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def filter(self, record):
        record.device = self.device
        return True


class _Log:
    def initialize(self, config: "GlobalConfig"):
        log_dir = get_run_output_dir(config)
        log_file = log_dir / "log.txt"

        device = config.device or "cpu"
        device_filter = _DeviceFilter(device)

        log_format = "%(asctime)s | %(device)-7s | %(name)-40s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(log_format, datefmt=date_format)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(device_filter)

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(device_filter)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        self.information("logging_configured", log_file=str(log_file))

    def for_source(self, source: str) -> _ContextLog:
        return _ContextLog(logging.getLogger(source))

    def for_context(self, **kwargs: Any) -> _ContextLog:
        return _ContextLog(logging.getLogger(), context=kwargs)

    def information(self, event: str, **kwargs: Any):
        logging.getLogger().info(
            self._format_msg(event, **kwargs), extra={"device": "n/a"}
        )

    def debug(self, event: str, **kwargs: Any):
        logging.getLogger().debug(
            self._format_msg(event, **kwargs), extra={"device": "n/a"}
        )

    def warning(self, event: str, **kwargs: Any):
        logging.getLogger().warning(
            self._format_msg(event, **kwargs), extra={"device": "n/a"}
        )

    def error(self, event: str, **kwargs: Any):
        logging.getLogger().error(
            self._format_msg(event, **kwargs), extra={"device": "n/a"}
        )

    def critical(self, event: str, **kwargs: Any):
        logging.getLogger().critical(
            self._format_msg(event, **kwargs), extra={"device": "n/a"}
        )

    def exception(self, event: str, **kwargs: Any):
        logging.getLogger().exception(
            self._format_msg(event, **kwargs), extra={"device": "n/a"}
        )

    def _format_msg(self, event: str, **kwargs: Any) -> str:
        if not kwargs:
            return event
        kv_pairs = [f"{k}={v}" for k, v in kwargs.items()]
        return f"{event} | {', '.join(kv_pairs)}"


Log = _Log()

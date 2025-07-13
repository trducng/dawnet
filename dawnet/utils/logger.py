import logging
import inspect
from typing import Literal


class Logger:
    def __init__(
        self,
        name=None,
        level: Literal["debug", "info", "warning", "error", "critical"] | int = "error",
        show_asctime: bool = True,
        show_name: bool = True,
        show_levelname: bool = True,
    ):
        self.logger = logging.getLogger(name or __name__)
        if isinstance(level, int):
            self.logger.setLevel(level)
        else:
            self.logger.setLevel(
                {
                    "debug": logging.DEBUG,
                    "info": logging.INFO,
                    "warning": logging.WARNING,
                    "error": logging.ERROR,
                    "critical": logging.CRITICAL
                }[level]
            )

        # Create console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = []
            if show_levelname: fmt.append("[%(levelname)s]")
            if show_asctime: fmt.append("%(asctime)s")
            if show_name: fmt.append("%(name)s")
            fmt.append("%(message)s")
            formatter = logging.Formatter(" ".join(fmt))
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _get_caller_info(self):
        """Get the class name and method name of the caller"""
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the actual caller
            caller_frame = frame.f_back.f_back.f_back

            # Get the method name
            method_name = caller_frame.f_code.co_name

            # Get the class name if it exists
            class_name = None
            if 'self' in caller_frame.f_locals:
                class_name = caller_frame.f_locals['self'].__class__.__name__
            elif 'cls' in caller_frame.f_locals:
                class_name = caller_frame.f_locals['cls'].__name__

            return class_name, method_name
        finally:
            del frame

    def _format_message(self, message):
        """Format message with class and method info"""
        class_name, method_name = self._get_caller_info()

        if class_name:
            return f"[{class_name}.{method_name}] {message}"
        else:
            return f"[{method_name}] {message}"

    def debug(self, message):
        formatted_msg = self._format_message(message)
        self.logger.debug(formatted_msg)

    def info(self, message):
        formatted_msg = self._format_message(message)
        self.logger.info(formatted_msg)

    def warning(self, message):
        formatted_msg = self._format_message(message)
        self.logger.warning(formatted_msg)

    def error(self, message):
        formatted_msg = self._format_message(message)
        self.logger.error(formatted_msg)

    def critical(self, message):
        formatted_msg = self._format_message(message)
        self.logger.critical(formatted_msg)


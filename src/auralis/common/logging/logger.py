#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import copy
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init()

VLLM_LOGGER_LEVEL = logging.INFO

class VLLMLogOverrider:
    """Override VLLM loggers to use custom formatting"""

    def __init__(self, target_logger: logging.Logger):
        self.target_logger = target_logger
        self.perf_pattern = re.compile(
            r"Avg prompt throughput:.+tokens/s,.+GPU KV cache usage:.+CPU KV cache usage:.+"
        )
        self.pipeline_warning_pattern = re.compile(r"Your model uses the legacy input pipeline instead of the new")
        self._override_vllm_loggers()

    def _override_vllm_loggers(self):
        """Override VLLM loggers to use our custom handler"""
        global VLLM_LOGGER_LEVEL
        for name in logging.root.manager.loggerDict:
            if name.startswith('vllm'):
                vllm_logger = logging.getLogger(name)
                current_level = VLLM_LOGGER_LEVEL
                vllm_logger.handlers.clear()
                vllm_logger.propagate = False
                handler = self._create_redirecting_handler()
                vllm_logger.addHandler(handler)
                vllm_logger.setLevel(current_level)

    def _create_redirecting_handler(self):
        """Create a handler that uses our custom formatting"""

        class RedirectHandler(logging.Handler):
            def __init__(self, target_logger, perf_pattern, pipe_warn):
                super().__init__()
                self.target_logger = target_logger
                self.pipe_warn = pipe_warn
                self.perf_pattern = perf_pattern

            def emit(self, record):
                msg = str(record.msg)
                if record.args:
                    msg = msg % record.args

                # Modify performance metrics format
                if self.perf_pattern.search(msg):
                    self.target_logger.log(record.levelno, f"Decoder performance: {msg}")
                elif self.pipe_warn.search(msg):
                    # Skip pipeline warning logs
                    pass
                else:
                    # Pass through all other logs normally
                    self.target_logger.log(record.levelno, msg)

        return RedirectHandler(self.target_logger, self.perf_pattern, self.pipeline_warning_pattern)


class ColoredFormatter(logging.Formatter):
    """Colored formatter with structured output and file location"""

    COLORS = {
        'DEBUG': {
            'color': Fore.CYAN,
            'style': Style.DIM,
            'icon': '🔍'
        },
        'INFO': {
            'color': Fore.GREEN,
            'style': Style.NORMAL,
            'icon': 'ℹ️'
        },
        'WARNING': {
            'color': Fore.YELLOW,
            'style': Style.BRIGHT,
            'icon': '⚠️'
        },
        'ERROR': {
            'color': Fore.RED,
            'style': Style.BRIGHT,
            'icon': '❌'
        },
        'CRITICAL': {
            'color': Fore.WHITE,
            'style': Style.BRIGHT,
            'bg': Back.RED,
            'icon': '💀'
        }
    }

    def format(self, record: logging.LogRecord) -> str:
        colored_record = copy.copy(record)

        # Get color scheme
        scheme = self.COLORS.get(record.levelname, {
            'color': Fore.WHITE,
            'style': Style.NORMAL,
            'icon': '•'
        })

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]

        # Get file location
        file_location = f"{os.path.basename(record.pathname)}:{record.lineno}"

        # Build components
        components = []

        # log formatting
        components.extend([
            f"{Fore.BLUE}{timestamp}{Style.RESET_ALL}",
            f"{Fore.WHITE}{Style.DIM}{file_location}{Style.RESET_ALL}",
            f"{scheme['color']}{scheme['style']}{scheme['icon']} {record.levelname:8}{Style.RESET_ALL}",
            f"{scheme['color']}{record.msg}{Style.RESET_ALL}"
        ])

        # Add exception info
        if record.exc_info:
            components.append(
                f"\n{Fore.RED}{Style.BRIGHT}"
                f"{''.join(traceback.format_exception(*record.exc_info))}"
                f"{Style.RESET_ALL}"
            )

        return " | ".join(components)


def setup_logger(
        name: Optional[Union[str, Path]] = None,
        level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a colored logger with VLLM override and file location

    Args:
        name: Logger name or __file__ for module name
        level: Logging level
    """
    # Get logger name from file path
    if isinstance(name, (str, Path)) and Path(name).suffix == '.py':
        name = Path(name).stem

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only add handler if none exists
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        # Override VLLM loggers to use our logger
        VLLMLogOverrider(logger)

    return logger


def set_vllm_logging_level(level: logging):
    """
    Set the logging level for VLLM loggers

    Args:
        level: Logging level to set (e.g., logging.INFO, logging.ERROR)
    """
    for name in logging.root.manager.loggerDict:
        if name.startswith('vllm'):
            vllm_logger = logging.getLogger(name)
            vllm_logger.setLevel(level)

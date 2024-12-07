import logging
import sys
from pathlib import Path
from datetime import datetime
import colorama
from colorama import Fore, Back, Style
from typing import Optional, Union
import re
import traceback
import copy
import os

# Initialize colorama
colorama.init()

VLLM_LOGGER_LEVEL = logging.INFO

class VLLMLogOverrider:
    """Override VLLM loggers to use custom formatting.
    
    This class intercepts and reformats logs from VLLM to provide consistent
    formatting and better integration with the application's logging system.
    It handles special cases like performance metrics and filters out unwanted
    pipeline warnings.

    Attributes:
        target_logger (logging.Logger): Logger to redirect VLLM logs to.
        perf_pattern (re.Pattern): Pattern to identify performance metric logs.
        pipeline_warning_pattern (re.Pattern): Pattern to identify pipeline warnings.
    """

    def __init__(self, target_logger: logging.Logger):
        """Initialize VLLM log overrider.

        Args:
            target_logger (logging.Logger): Logger to redirect VLLM logs to.
        """
        self.target_logger = target_logger
        self.perf_pattern = re.compile(
            r"Avg prompt throughput:.+tokens/s,.+GPU KV cache usage:.+CPU KV cache usage:.+"
        )
        self.pipeline_warning_pattern = re.compile(r"Your model uses the legacy input pipeline instead of the new")
        self._override_vllm_loggers()

    def _override_vllm_loggers(self):
        """Override all VLLM loggers to use custom formatting.
        
        This method finds all loggers with names starting with 'vllm' and
        replaces their handlers with our custom handler.
        """
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
        """Create a custom logging handler for VLLM logs.
        
        Returns:
            logging.Handler: Handler that reformats and redirects VLLM logs.
        """

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
    """Colored formatter for structured log output.
    
    This formatter adds color-coding, icons, timestamps, and file location
    information to log messages. It supports different color schemes for
    different log levels and includes special formatting for exceptions.

    !!! tip "Color Scheme"
        Each log level has its own color scheme and icon:

        - DEBUG: Cyan (dim) 🔍
        - INFO: Green ℹ️
        - WARNING: Yellow (bright) ⚠️
        - ERROR: Red (bright) ❌
        - CRITICAL: White on Red background 💀

    !!! example "Sample Output"
        ```
        10:30:45.123 | app.py:42 | ℹ️ INFO     | Starting application
        10:30:45.234 | model.py:156 | ⚠️ WARNING  | GPU memory running low
        ```

    See Also:
        - [`setup_logger`][auralis.common.logging.logger.setup_logger]: Main logger setup function
        - [`VLLMLogOverrider`][auralis.common.logging.logger.VLLMLogOverrider]: VLLM log handler

    Attributes:
        COLORS (dict): Color schemes for different log levels, including:
            - color: Foreground color
            - style: Text style (dim, normal, bright)
            - icon: Emoji icon for the log level
            - bg: Background color (for critical logs)
    """

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
        """Format a log record with color and structure.

        This method formats log records with:
        - Timestamp in HH:MM:SS.mmm format
        - File location (filename:line)
        - Color-coded level name with icon
        - Color-coded message
        - Formatted exception traceback if present

        Args:
            record (logging.LogRecord): Log record to format.

        Returns:
            str: Formatted log message with color and structure.
        """
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
    """Set up a colored logger with VLLM override.

    This function creates or retrieves a logger with colored output and
    automatic VLLM log interception. If a file path is provided as the name,
    it will use the filename (without extension) as the logger name.

    !!! note "VLLM Integration"
        When used with VLLM components, this logger automatically:
        
        - Intercepts and reformats VLLM logs
        - Filters redundant pipeline warnings
        - Enhances performance metric visibility

    !!! example "Basic Usage"
        ```python
        # Setup with module name
        logger = setup_logger(__name__)
        logger.info("Starting process")

        # Setup with custom name and level
        debug_logger = setup_logger("debug_logs", logging.DEBUG)
        debug_logger.debug("Detailed information")
        ```

    Args:
        name (Optional[Union[str, Path]], optional): Logger name or __file__ for
            module name. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.

    See Also:
        - [`ColoredFormatter`][auralis.common.logging.logger.ColoredFormatter]: Formatter class
        - [`VLLMLogOverrider`][auralis.common.logging.logger.VLLMLogOverrider]: VLLM integration
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
    """Set logging level for all VLLM loggers.

    This function finds all loggers with names starting with 'vllm' and
    sets their logging level. This is useful for controlling the verbosity
    of VLLM's output.

    Args:
        level (logging): Logging level to set (e.g., logging.INFO, logging.DEBUG).

    Example:
        >>> set_vllm_logging_level(logging.WARNING)  # Reduce VLLM verbosity
    """
    for name in logging.root.manager.loggerDict:
        if name.startswith('vllm'):
            vllm_logger = logging.getLogger(name)
            vllm_logger.setLevel(level)

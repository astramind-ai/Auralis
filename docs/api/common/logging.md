# Logging

The logging module provides a flexible and colorful logging system with built-in VLLM integration.

## Overview

The module consists of three main components:

1. `ColoredFormatter`: Formats log messages with colors and icons
2. `VLLMLogOverrider`: Integrates VLLM logging with Auralis
3. `setup_logger`: Main function to configure logging

## ColoredFormatter

::: auralis.common.logging.logger.ColoredFormatter
    options:
      show_root_heading: true
      show_source: true

## VLLMLogOverrider

::: auralis.common.logging.logger.VLLMLogOverrider
    options:
      show_root_heading: true
      show_source: true

## Functions

### setup_logger

::: auralis.common.logging.logger.setup_logger
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Basic Logging

```python
from auralis.common.logging import setup_logger

# Create logger
logger = setup_logger(__name__)

# Use different log levels
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Custom Configuration

```python
import logging
from auralis.common.logging import setup_logger

# Create debug logger
logger = setup_logger(
    name="debug_logger",
    level=logging.DEBUG
)

# Log with context
logger.debug("Processing file", extra={"file": "example.txt"})
```

### VLLM Integration

The logger automatically handles VLLM logs:

```python
from auralis.common.logging import setup_logger

# Create logger
logger = setup_logger(__name__)

# VLLM logs will be automatically formatted
# Example output:
# 10:30:45.123 | model.py:42 | ℹ️ INFO | Avg prompt throughput: 123.4 tokens/s
```

## Best Practices

!!! tip "Logging Tips"
    1. Use appropriate log levels:
        - DEBUG: Detailed information for debugging
        - INFO: General operational messages
        - WARNING: Warning messages for potential issues
        - ERROR: Error messages for actual problems
        - CRITICAL: Critical errors that require immediate attention

    2. Include context in log messages:
        ```python
        logger.info(f"Processing file {filename}")
        logger.error("Failed to load model", exc_info=True)
        ```

    3. Use structured logging when possible:
        ```python
        logger.info("Model loaded", extra={
            "model_name": "xtts_v2",
            "device": "cuda:0"
        })
        ``` 
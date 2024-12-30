import time
from dataclasses import dataclass, field
from functools import wraps
from typing import TypeVar, AsyncGenerator, Callable, Any

from auralis.common.definitions.metrics.performance import TTSMetricsTracker
from auralis.common.logging.logger import setup_logger


metrics = TTSMetricsTracker()

T = TypeVar('T')


def track_generation(func: Callable[..., AsyncGenerator[T, None]]) -> Callable[..., AsyncGenerator[T, None]]:
    """Decorator to track TTS generation performance metrics.

    This decorator wraps TTS generation functions to automatically track
    performance metrics for each generated audio chunk. It updates the global
    metrics tracker and logs performance statistics at regular intervals.

    Args:
        func (Callable[..., AsyncGenerator[T, None]]): Async generator function
            that yields TTS outputs.

    Returns:
        Callable[..., AsyncGenerator[T, None]]: Wrapped function that tracks metrics.

    Example:
        >>> @track_generation
        ... async def generate_speech(text: str) -> AsyncGenerator[TTSOutput, None]:
        ...     # Generation code here
        ...     yield output
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> AsyncGenerator[T, None]:
        """Wrapped generation function that tracks metrics.

        Args:
            *args: Positional arguments passed to the generation function.
            **kwargs: Keyword arguments passed to the generation function.

        Yields:
            T: TTS output chunks with tracked metrics.
        """
        async for output in func(*args, **kwargs):
            if output.start_time:
                audio_seconds = output.array.shape[0] / output.sample_rate

                if metrics.update_metrics(output.token_length, audio_seconds):
                    metrics.logger.info(
                        f"Generation metrics | "
                        f"Throughput: {metrics.requests_per_second:.2f} req/s | "
                        f"{metrics.tokens_per_second:.1f} tokens/s | "
                        f"Latency: {metrics.ms_per_second_of_audio:.0f}ms per second of audio generated"
                    )
                    metrics.reset_window()
            yield output

    return wrapper
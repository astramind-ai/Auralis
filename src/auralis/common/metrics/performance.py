import time
from dataclasses import dataclass, field
from functools import wraps
from typing import TypeVar, AsyncGenerator, Callable, Any
from auralis.common.logging.logger import setup_logger


T = TypeVar('T')


@dataclass
class TTSMetricsTracker:
    """Performance metrics tracker for TTS generation.
    
    This class tracks and calculates various performance metrics for TTS generation,
    including throughput (requests and tokens per second) and latency. It maintains
    a sliding window of metrics and provides periodic logging.

    Attributes:
        window_start (float): Start time of current metrics window.
        last_log_time (float): Time of last metrics log.
        log_interval (float): Seconds between metric logs.
        window_tokens (int): Total tokens processed in current window.
        window_audio_seconds (float): Total audio seconds generated in window.
        window_requests (int): Total requests processed in window.
    """

    logger = setup_logger(__file__)

    window_start: float = field(default_factory=time.time)
    last_log_time: float = field(default_factory=time.time)
    log_interval: float = 5.0  # sec between logs

    window_tokens: int = 0
    window_audio_seconds: float = 0
    window_requests: int = 0

    @property
    def requests_per_second(self) -> float:
        """Calculate requests processed per second.

        Returns:
            float: Average requests per second in current window.
        """
        elapsed = time.time() - self.window_start
        return self.window_requests / elapsed if elapsed > 0 else 0

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens processed per second.

        Returns:
            float: Average tokens per second in current window.
        """
        elapsed = time.time() - self.window_start
        return self.window_tokens / elapsed if elapsed > 0 else 0

    @property
    def ms_per_second_of_audio(self) -> float:
        """Calculate processing time per second of generated audio.

        Returns:
            float: Milliseconds required to generate one second of audio.
        """
        elapsed = (time.time() - self.window_start) * 1000  # in ms
        return elapsed / self.window_audio_seconds if self.window_audio_seconds > 0 else 0

    def reset_window(self) -> None:
        """Reset all metrics for a new window.
        
        This method resets all counters and timestamps to start a fresh
        metrics collection window.
        """
        current_time = time.time()
        self.last_log_time = current_time
        # reset window
        self.window_start = current_time
        self.window_tokens = 0
        self.window_audio_seconds = 0
        self.window_requests = 0

    def update_metrics(self, tokens: int, audio_seconds: float) -> bool:
        """Update metrics with new generation results.

        Args:
            tokens (int): Number of tokens processed.
            audio_seconds (float): Seconds of audio generated.

        Returns:
            bool: Whether metrics should be logged based on log interval.
        """
        self.window_tokens += tokens
        self.window_audio_seconds += audio_seconds
        self.window_requests += 1

        current_time = time.time()
        should_log = current_time - self.last_log_time >= self.log_interval

        return should_log


metrics = TTSMetricsTracker()


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
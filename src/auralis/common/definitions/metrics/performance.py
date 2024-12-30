import time
from dataclasses import dataclass, field
from functools import wraps
from typing import TypeVar, AsyncGenerator, Callable, Any
from auralis.common.logging.logger import setup_logger


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

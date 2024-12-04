#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

from functools import wraps
from typing import TypeVar, AsyncGenerator, Callable

from auralis.common.definitions.metrics.tracker import TTSMetricsTracker

T = TypeVar('T')


metrics = TTSMetricsTracker()


def track_generation(func: Callable[..., AsyncGenerator[T, None]]) -> Callable[..., AsyncGenerator[T, None]]:
    @wraps(func)
    async def wrapper(*args, **kwargs) -> AsyncGenerator[T, None]:
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
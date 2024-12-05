#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio.timeouts
from typing import Optional, List, Any, Callable, Union

from async_timeout import timeout

from auralis import TTSRequest, setup_logger
from auralis.common.definitions.scheduler.context import GenerationContext

logger = setup_logger(__name__)

class AsyncDynamicBatcher:
    """
    Dynamic batcher for async functions
    """
    def __init__(self, fn: Callable[[List[Any]], Any], max_wait: float = 0.1, has_vllm: bool= False):
        self.max_size = None # we'll wait to profile this
        self.is_vllm = has_vllm
        self.max_wait = max_wait
        self.process = fn
        # inspect if the fn is async
        if not asyncio.iscoroutinefunction(fn):
            raise ValueError(f"{fn.__name__} must be an async function to not block the event loop")

    def set_max_batch_size(self, max_batch_size):
        logger.debug(f"Setting max batch size to {max_batch_size} for function {self.process.__name__}")
        self.max_size = max_batch_size

    async def create_batch(self,
                           queue: asyncio.Queue[Union[TTSRequest, GenerationContext]]
                           ) -> Optional[TTSRequest, GenerationContext]:
        """Create batch from queue items, considering size and wait time"""
        if queue.empty():
            return None

        items = []
        try:
            # Get first item immediately
            items.append(await queue.get())
            if self.is_vllm:
                return items
            # Try to get more items up to max_size or timeout
            async with timeout(self.max_wait):
                while len(items) < self.max_size:
                    if queue.empty():
                        break
                    items.append(await queue.get())

        except TimeoutError:
            pass  # We'll process whatever we got

        return items if items else None

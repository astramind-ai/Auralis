#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio.timeouts
from queue import Queue
from typing import Optional, List, Any, Callable

from async_timeout import timeout


class AsyncDynamicBatcher:
    # TODO: make so that if the model has a vllm component the batchsize will be 1 since
    #  we don't actually batch vllm request and as soon as they're ready they need to exit the queue
    def __init__(self, fn: Callable[[List[Any]], Any], max_wait: float = 0.1):
        self.max_size = None # we'll wait to profile this
        self.max_wait = max_wait
        self.process = fn
        # inspect if the fn is async
        if not asyncio.iscoroutinefunction(fn):
            raise ValueError(f"{fn.__name__} must be an async function to not block the event loop")

    def set_max_batch_size(self, max_batch_size):
        self.max_size = max_batch_size

    async def create_batch(self, queue: Queue) -> Optional[Any]:
        """Create batch from queue items, considering size and wait time"""
        if queue.empty():
            return None

        items = []
        try:
            # Get first item immediately
            items.append(await queue.get())

            # Try to get more items up to max_size or timeout
            async with timeout(self.max_wait):
                while len(items) < self.max_size:
                    if queue.empty():
                        break
                    items.append(await queue.get())

        except TimeoutError:
            pass  # We'll process whatever we got

        return items if items else None

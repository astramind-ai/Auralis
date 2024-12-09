#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.
import asyncio
from collections.abc import Coroutine
from typing import Optional, List, Any, Callable, Union, AsyncGenerator, Tuple

from async_timeout import timeout
import inspect

from auralis import TTSRequest, setup_logger, TTSOutput
from auralis.common.definitions.batch.batchable_item import BatchableItem
from auralis.common.definitions.batch.batches import BatchedItems
from auralis.common.definitions.scheduler.context import GenerationContext
from auralis.common.definitions.types.orchestrator import BatcherFunction

logger = setup_logger(__name__)

# TODO rimuover eil dinamiyc batcher e fare un compoennte che lavory com eun lock avanzato, 
class AsyncDynamicBatcher:
    """
    Dynamic batcher for async functions
    """
    def __init__(self,
                 fn: BatcherFunction,
                 max_size: int = 1,
                 max_concurrency: int = 1,
                 max_wait: float = 0.1,
                 has_vllm: bool= False
                 ):
        self.max_batch_size = (max_size or 0) * max_concurrency
        self.is_vllm = has_vllm
        self.max_wait = max_wait
        self.process = fn
        # inspect if the fn is async ( or an async generator )
        if not (inspect.isasyncgenfunction(fn) or inspect.iscoroutinefunction(fn)):
            raise ValueError(f"{fn.__name__} must be an async function to not block the event loop")

    #def set_max_batch_size(self, max_batch_size): FIXME(mlinmg) this will be needed when memory blocks are added
    #    logger.debug(f"Setting max batch size to {max_batch_size} for function {self.process.__name__}")
    #    self.max_size = max_batch_size

    async def create_batch(self,
                           queue: asyncio.Queue[BatchableItem]
                           ) -> Tuple[Optional[BatchedItems], Optional[asyncio.Queue[Any]]]:
        """Create batch from queue items, considering size and wait time"""
        if queue.empty():
            return None

        items = BatchedItems(self.process.__name__)
        output_queue = None  # Inizializza output_queue a None

        try:
            # Get first item immediately
            item, output_queue = await queue.get()
            items.batch(await item)

            if self.is_vllm:
                return items, output_queue  # Return immediately output queue

            # Try to get more items up to max_size or timeout
            async with timeout(self.max_wait):

                while items.length < self.max_batch_size:
                    if queue.empty():
                        break
                    item, _ = await queue.get()  # output_queue should be None
                    items.batch(item)

        except TimeoutError:
            pass  # We'll process whatever we got

        return items, output_queue if items else None  # return output queue
import asyncio
from typing import Callable

from auralis import setup_logger
from auralis.common.scheduling.dynamic_resource_lock import DynamicResourceLock

logger = setup_logger(__name__)
class AsyncScheduler:
    def __init__(self, resource_lock: DynamicResourceLock, processing_function: Callable, stage_name: str):
        self.resource_lock = resource_lock
        self.processing_function = processing_function
        self.stage_name = stage_name
        self.input_queue = asyncio.Queue()

    def get_next_stage(self, stage: str):
        stages = ['conditioning', 'phonetic', 'speech']
        return stages[stages.index(stage) + 1]

    async def process(self, orchestrator: 'Orchestrator'):
        """Processes items, respecting resource limits and signaling completion."""
        while True:
            request_id, input_data, stage, output, completion_event = await self.input_queue.get()

            batch_size =input_data.length(self.stage_name)
            try:
                async with self.resource_lock.lock_resource(batch_size):
                    new_output = await self.processing_function(input_data)

                    next_stage = self.get_next_stage(stage)
                    if next_stage == "completed":
                        completion_event.set()

                    # Put the item back into the main queue with updated stage and output
                    await orchestrator.queue.put((request_id, input_data, next_stage, new_output, completion_event))
            except Exception as e: # TODO: better except
                logger.error(f"Error processing request {request_id}: {e}")

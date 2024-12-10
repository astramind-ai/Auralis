import asyncio
from typing import Callable

from auralis import setup_logger
from auralis.common.scheduling.dynamic_resource_lock import DynamicResourceLock

logger = setup_logger(__name__)
class AsyncScheduler:
    def __init__(self, resource_lock: DynamicResourceLock, processing_function: Callable, stage_name: str):
        """
        Initialize an AsyncScheduler.

        Parameters
        ----------
        resource_lock : DynamicResourceLock
            The lock that controls the resources available to the scheduler.
        processing_function : Callable
            The function that will be called to process each item in the queue.
        stage_name : str
            The name of the stage that this scheduler represents.
        """
        self.resource_lock = resource_lock
        self.processing_function = processing_function
        self.stage_name = stage_name
        self.input_queue = asyncio.Queue()

    def get_next_stage(self, stage: str):
        stages = ['conditioning', 'phonetic', 'speech']
        return stages[stages.index(stage) + 1]

    async def process(self, orchestrator: 'Orchestrator'):
        """
        Process the items in the input queue.

        This method runs indefinitely, pulling items from the input queue and
        processing them with the provided processing function. The output of the
        processing function is then put back into the main queue with the next
        stage and output.

        Parameters
        ----------
        orchestrator : Orchestrator
            The orchestrator that owns this scheduler.

        Returns
        -------
        None
        """
        while True:
            request_id, input_data, stage, output, completion_event = await self.input_queue.get()

            batch_size = input_data.length(self.stage_name)
            try:
                async with self.resource_lock.lock_resource(batch_size):
                    new_outputs = await self.processing_function(input_data)

                    next_stage = self.get_next_stage(stage)
                    if next_stage == "completed":
                        completion_event.set()

                    # Put the item back into the main queue with updated stage and output
                    # here since we have multiple new_output(it is a list)
                    # we have to put all of them in the queue(in order)
                    [
                        await orchestrator.queue.put(
                            (request_id, new_output, next_stage, new_output, completion_event)
                        ) for new_output in new_outputs
                    ]
            except Exception as e: # TODO: better except
                logger.error(f"Error processing request {request_id}: {e}")

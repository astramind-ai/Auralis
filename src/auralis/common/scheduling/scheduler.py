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
        stages = ['conditioning', 'phonetic', 'synthesis', 'completed']
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
            input_data, stage, completion_event = await self.input_queue.get()

            try:
                async with self.resource_lock.lock_resource(input_data.length):
                    new_outputs = await self.processing_function(input_data)

                    next_stage = self.get_next_stage(stage)
                    if next_stage == "completed":
                        completion_event.set()

                    # Put the item back into the main queue with updated stage and output
                    # here since we have multiple new_output(it is a list)
                    # we have to put all of them in the queue(in order)
                    if isinstance(new_outputs, list):
                        [
                        await orchestrator.queue.put(
                            (new_output, next_stage, completion_event)
                        ) for new_output in new_outputs
                    ]
                    else:
                        await orchestrator.queue.put(
                            (new_outputs, next_stage, completion_event)
                        )
            except Exception as e: # TODO: better except
                logger.error(f"Error processing request {input_data.request_id}: {e}")

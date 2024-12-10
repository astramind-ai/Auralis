import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from typing import Callable

from auralis import TTSOutput, setup_logger
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.metrics.performance import track_generation
from auralis.common.scheduling.dynamic_resource_lock import DynamicResourceLock
from auralis.common.scheduling.profiler import Profiler
from auralis.common.scheduling.scheduler import AsyncScheduler
from auralis.models.base import BaseAsyncTTSEngine

logger = setup_logger(__name__)

class Orchestrator:
    def __init__(self, engine: BaseAsyncTTSEngine):
        """
        Initialize the Orchestrator with the given TTS engine.

        This constructor sets up the necessary schedulers and profiling for
        the various phases of text-to-speech processing, including conditioning,
        phonetics, and synthesis. It configures these schedulers based on the
        engine's configuration, which includes concurrency settings and maximum sizes.

        Args:
            engine (BaseAsyncTTSEngine): The engine used for TTS processing. It
                provides the functions for each processing phase and configuration
                details.

        Attributes:
            schedulers (list): A list of AsyncScheduler instances for managing
                different phases of TTS processing.
            preprocessing_phase_fn (Callable): Function to preprocess input data.
            queue (asyncio.Queue): Queue to manage requests for processing.
            scheduler_tasks (list): List of tasks handling scheduling.
            processing_task (Optional[asyncio.Task]): Task for processing the queue.
        """
        conditioning_phase_fn = engine.conditioning_phase
        phonetics_phase_fn = engine.phonetic_phase
        synthesis_phase_fn = engine.speech_phase
        eng_config = engine.config

        fake_data_factories = engine.info['fake_data_factories']
        concurrences = eng_config['concurrences']
        max_sizes = eng_config['max_sizes']

        self.schedulers = [
            AsyncScheduler(
                DynamicResourceLock(max_sizes[0] * concurrences[0]),
                conditioning_phase_fn,
                "conditioning"
            ),
            AsyncScheduler(
                DynamicResourceLock(max_size=-1),
                phonetics_phase_fn,
                "phonetic"
            ),
            AsyncScheduler(
                DynamicResourceLock(max_sizes[2] * concurrences[2]),
                synthesis_phase_fn,
                "synthesis"
            )
        ]
        self.preprocessing_phase_fn = engine.preprocess_inputs

        Profiler.profile(
            fake_data_factories,
            (conditioning_phase_fn, phonetics_phase_fn, synthesis_phase_fn),
            eng_config
        )

        self.queue = asyncio.Queue()  # Single queue for all requests
        self.scheduler_tasks = []
        self.processing_task = None  # Task for processing the queue

    async def start_schedulers(self):
        self.scheduler_tasks = [
            asyncio.create_task(s.process(self)) for s in self.schedulers
        ]

    async def run(self, request: TTSRequest) -> AsyncGenerator[TTSOutput, None]:
        if not self.scheduler_tasks:
            await self.start_schedulers()

        request_id = request.request_id
        input_data = await self.preprocessing_phase_fn(request)
        completion_event = asyncio.Event()
        await self.queue.put((request_id, input_data, "conditioning", None, completion_event))

        # Start processing the queue if not already started
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self.process_queue())

        while True:
            if completion_event.is_set():
                # Find the completed item in the queue
                for item in self.queue._queue:
                    if item[0] == request_id and item[2] == "completed":
                        logger.info(f"Request {request_id} finished")
                        if isinstance(item[3], AsyncGenerator):
                            async for output_item in item[3]:
                                yield output_item  # Yield output items
                        else:
                            yield item[3]  # Yield the output
                        # Remove the completed item from the queue
                        self.queue._queue.remove(item)
                        self.queue.task_done()
                        break # Exit the inner loop after yielding the output for the completed request
            else:
                await asyncio.sleep(0)

    async def process_queue(self):
        while True:
            # Get the next item from the queue (non-blocking)
            request_id, input_data, stage, output, completion_event = await self.queue.get()

            if stage != "completed":
                # Find the appropriate scheduler and put the item in its input queue
                for scheduler in self.schedulers:
                    if scheduler.stage_name == stage:
                        await scheduler.input_queue.put((request_id, input_data, stage, output, completion_event))
                        break  # Important: break out of the inner loop after forwarding the item
                else:  # This 'else' clause is associated with the 'for' loop
                    # If no scheduler was found for the current stage, put the item back in the queue
                    await self.queue.put((request_id, input_data, stage, output, completion_event))

            else:
                # If completed, put it back for the run() method to pick up
                await self.queue.put((request_id, input_data, stage, output, completion_event))

            self.queue.task_done()

    async def shutdown(self):
        if self.processing_task:
            self.processing_task.cancel()
            await self.processing_task

        for task in self.scheduler_tasks:
            task.cancel()
        await asyncio.gather(*self.scheduler_tasks, return_exceptions=True)
#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio
from collections.abc import AsyncGenerator
from typing import Callable, Tuple, Optional

from auralis import TTSOutput, setup_logger
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.definitions.types.scheduler import FakeFactoriesForSchedulerProfiling
from auralis.common.metrics.performance import track_generation
from auralis.common.scheduling.dynamic_batcher import AsyncDynamicBatcher
from auralis.common.scheduling.scheduler import AsyncScheduler

logger = setup_logger(__name__)


class Orchestrator:
    def __init__(self,
                 conditioning_phase_fn: Callable,
                 phonetics_phase_fn: Callable,
                 synthesis_phase_fn: Callable,
                 fake_data_factories: FakeFactoriesForSchedulerProfiling):

        self.schedulers = [
            AsyncScheduler(AsyncDynamicBatcher(conditioning_phase_fn)),
            AsyncScheduler(AsyncDynamicBatcher(phonetics_phase_fn, fake_data_factories[1] is None)), # this means the fonetic phase is managed by vllm
            AsyncScheduler(AsyncDynamicBatcher(synthesis_phase_fn))
        ]

        self.profile(fake_data_factories)

        # Start scheduler processing tasks
        self.scheduler_tasks = [
            asyncio.create_task(s.process())
            for s in self.schedulers
        ]

    def profile(self, fake_data_factories: FakeFactoriesForSchedulerProfiling):
        logger.info(f"Starting Auralis profiling...")
        raise NotImplementedError

    @track_generation
    async def _track_yield(self, yieldable: AsyncGenerator[TTSOutput, None]):
        yield yieldable

    async def run(self, request: TTSRequest) -> AsyncGenerator[TTSOutput, None]:
        """Main entry point - clients submit items and get results"""
        logger.info(f"Starting request {request.request_id}")
        await self.schedulers[0].input_queue.put(request)

        # Connect pipeline stages and yield results
        while True:
            for i in range(len(self.schedulers) - 1):
                batch = await self.schedulers[i].output_queue.get()

                await self.schedulers[i + 1].input_queue.put(*batch if isinstance(batch, list ) else batch)

            # Yield results from final stage
            final_batch = await self.schedulers[-1].output_queue.get()
            async for tracked_item in self._track_yield(final_batch):
                if tracked_item.is_finished:
                     logger.info(f"Request {tracked_item.request_id} finished in {tracked_item.duration} seconds")
                yield tracked_item

    async def shutdown(self):
        for task in self.scheduler_tasks:
            task.cancel()
        await asyncio.gather(*self.scheduler_tasks, return_exceptions=True)
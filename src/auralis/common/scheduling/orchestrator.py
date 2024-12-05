#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio
from collections.abc import AsyncGenerator
from typing import Callable, Tuple, Optional

import torch

from auralis import TTSOutput, setup_logger
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.metrics.performance import track_generation
from auralis.common.scheduling.dynamic_batcher import AsyncDynamicBatcher
from auralis.common.scheduling.profiler import Profiler
from auralis.common.scheduling.scheduler import AsyncScheduler
from auralis.models.base import BaseAsyncTTSEngine

logger = setup_logger(__name__)


class Orchestrator:
    def __init__(self,
                 engine: BaseAsyncTTSEngine):
        conditioning_phase_fn = engine.conditioning_phase
        phonetics_phase_fn = engine.phonetic_phase
        synthesis_phase_fn = engine.speech_phase
        fake_data_factories = engine.info['fake_data_factories']
        eng_config = engine.config
        concurrences = eng_config['concurrences']
        self.schedulers = [
            AsyncScheduler(AsyncDynamicBatcher(conditioning_phase_fn, max_size=concurrences[0])),
            AsyncScheduler(AsyncDynamicBatcher(phonetics_phase_fn, max_size=concurrences[1], has_vllm=fake_data_factories[1] is None)), # this means the fonetic phase is managed by vllm
            AsyncScheduler(AsyncDynamicBatcher(synthesis_phase_fn, max_size=concurrences[2]))
        ]

        Profiler.profile(fake_data_factories,
                             (conditioning_phase_fn, phonetics_phase_fn, synthesis_phase_fn), eng_config)


        #self.memory_manager = AuralisMemoryManager( #TODO This will need a much deeper implementation (help needed)
        #    memory_shapes,
        #    memory_requirements,
        #    dtype=dtype
        #)
        # Start scheduler processing tasks
        self.scheduler_tasks = []

    async def start_schedulers(self):
        self.scheduler_tasks = [
            asyncio.create_task(s.process()) for s in self.schedulers
        ]

    @track_generation
    async def _track_yield(self, yieldable: AsyncGenerator[TTSOutput, None]):
        yield yieldable

    async def run(self, request: TTSRequest) -> AsyncGenerator[TTSOutput, None]:
        """Main entry point - clients submit items and get results"""
        if len(self.scheduler_tasks) == 0:
            await self.start_schedulers()
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
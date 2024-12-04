#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio
from collections.abc import AsyncGenerator
from typing import Callable

from auralis import TTSOutput
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.scheduling.dynamic_batcher import AsyncDynamicBatcher
from auralis.common.scheduling.scheduler import AsyncScheduler


class Orchestrator:
    def __init__(self, conditioning_phase_fn: Callable, phonetics_phase_fn: Callable, synthesis_phase_fn: Callable):
        self.schedulers = [
            AsyncScheduler(AsyncDynamicBatcher(conditioning_phase_fn)),
            AsyncScheduler(AsyncDynamicBatcher(phonetics_phase_fn)),
            AsyncScheduler(AsyncDynamicBatcher(synthesis_phase_fn))
        ]

        # Start scheduler processing tasks
        self.scheduler_tasks = [
            asyncio.create_task(s.process())
            for s in self.schedulers
        ]

    async def run(self, request: TTSRequest) -> AsyncGenerator[TTSOutput, None]:
        """Main entry point - clients submit items and get results"""
        await self.schedulers[0].input_queue.put(request)

        # Connect pipeline stages and yield results
        while True:
            for i in range(len(self.schedulers) - 1):
                batch = await self.schedulers[i].output_queue.get()

                await self.schedulers[i + 1].input_queue.put(*batch if isinstance(batch, list ) else batch)

            # Yield results from final stage
            final_batch = await self.schedulers[-1].output_queue.get()
            yield final_batch

    async def shutdown(self):
        for task in self.scheduler_tasks:
            task.cancel()
        await asyncio.gather(*self.scheduler_tasks, return_exceptions=True)
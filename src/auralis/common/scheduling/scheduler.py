#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio
from typing import Union, AsyncGenerator

from auralis import TTSRequest
from auralis.common.definitions.scheduler.context import GenerationContext


class AsyncScheduler:
    def __init__(self, batcher, is_streaming=False):
        self.batcher = batcher
        self.is_streaming = is_streaming
        self.input_queue: asyncio.Queue[Union[TTSRequest, GenerationContext]] = asyncio.Queue()
        self.output_queue: asyncio.Queue[[Union[GenerationContext, AsyncGenerator[TTSRequest, None]]]]= asyncio.Queue()

    async def process(self):
        while True:
            batch = await self.batcher.create_batch(self.input_queue)
            if not batch:
                continue

            if self.is_streaming:
                # The last scheduler is a streaming one
                async for result in self.batcher.process(batch):
                    await self.output_queue.put(result)
            else:
                # the first two schedulers are non-streaming
                processed = await self.batcher.process(batch)
                await self.output_queue.put(processed)





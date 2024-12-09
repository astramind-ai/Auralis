#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio
import json
import logging
import os
import queue
import threading
import time
import uuid
from functools import partial
from typing import AsyncGenerator, Optional, Dict, Union, Generator, List

from auralis.common.scheduling.orchestrator import Orchestrator
from huggingface_hub import hf_hub_download

from auralis.common.definitions.dto.output import TTSOutput
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.logging.logger import setup_logger, set_vllm_logging_level
from auralis.common.metrics.performance import track_generation
from auralis.models.base import BaseAsyncTTSEngine

logger = setup_logger(__file__)

class TTS:
    def __init__(self, scheduler_max_concurrency: int = None, vllm_logging_level=logging.DEBUG):
        set_vllm_logging_level(vllm_logging_level)

        self.orchestrator: Optional[Orchestrator] = None
        self.tts_engine: Optional[BaseAsyncTTSEngine] = None
        if scheduler_max_concurrency is not None:
            logger.warning("scheduler_max_concurrency passed as a TTS argument is deprecated and will be removed "
                           "in future releases, please pass it to the method from_pretrained")
        self.concurrency = scheduler_max_concurrency # kept for backwards compatibility
        self.logger = setup_logger(__file__)

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        self._async = None

    @staticmethod
    def split_requests(request: TTSRequest, max_length: int = 100000) -> List[TTSRequest]:
        """Split a request into multiple chunks."""
        if len(request.text) <= max_length:
            return [request]

        text_chunks = [request.text[i:i + max_length]
                       for i in range(0, len(request.text), max_length)]

        return [
            (copy := request.copy(), setattr(copy, 'text', chunk), setattr(copy, 'request_id', uuid.uuid4().hex))[0]
            for chunk in text_chunks
        ]

    def _start_orchestrator(self):
        self.orchestrator = Orchestrator(self.tts_engine)

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _non_streaming_sync_wrapper(self, requests):
        # Use asyncio.run_coroutine_threadsafe and get a concurrent.futures.Future
        future = asyncio.run_coroutine_threadsafe(self._process_requests(requests), self.loop)
        try:
            return future.result()  # This will block until the result is available
        except Exception as e:
            raise e

    def _streaming_sync_wrapper(self, requests):
        q = queue.Queue()

        async def produce():
            try:
                for sub_request in requests:
                    async for chunk in self.orchestrator.run(
                            request=sub_request,
                    ):
                        q.put(chunk)
                q.put(None)  # Signal completion
            except Exception as e:
                q.put(e)
                q.put(None)  # Ensure the generator exits

        # Schedule the coroutine in the dedicated event loop
        future = asyncio.run_coroutine_threadsafe(produce(), self.loop)

        # Return a generator that yields items from the queue
        def sync_generator():
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

            # Ensure that the coroutine has completed
            try:
                future.result()
            except Exception as e:
                raise e

        return sync_generator()

    async def _process_requests(self, requests):
        chunks = []
        for sub_request in requests:
            async for chunk in self.orchestrator.run(
                    request=sub_request,
            ):
                chunks.append(chunk)
        return TTSOutput.combine_outputs(chunks)

    async def _process_multiple_requests(self,
                                         requests: List[TTSRequest],
                                         results: Optional[List] = None) -> Optional[TTSOutput]:
        """Process multiple requests in parallel."""
        output_queues = [asyncio.Queue() for _ in requests] if results is not None else None

        async def process_subrequest(idx, sub_request, queue: Optional[asyncio.Queue] = None):
            chunks = []
            async for chunk in self.orchestrator.run(
                    request=sub_request,

            ):
                chunks.append(chunk)
                if queue is not None:
                    await queue.put(chunk)

            if queue is not None:
                await queue.put(None)
            return chunks

        tasks = [
            asyncio.create_task(
                process_subrequest(
                    idx,
                    sub_request,
                    output_queues[idx] if output_queues else None
                )
            )
            for idx, sub_request in enumerate(requests)
        ]

        if results is not None:
            for idx, queue in enumerate(output_queues):
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    results[idx].append(chunk)
            return None
        else:
            all_chunks = await asyncio.gather(*tasks)
            complete_audio = [chunk for chunks in all_chunks for chunk in chunks]
            return TTSOutput.combine_outputs(complete_audio)

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        """Load a pretrained model."""
        from auralis.models.registry import ModelRegistry

        try:
            with open(os.path.join(model_name_or_path, 'config.json'), 'r') as f:
                config = json.load(f)

        except FileNotFoundError:
            try:
                config_path = hf_hub_download(repo_id=model_name_or_path, filename='config.json')
                with open(config_path, 'r') as f:
                    config = json.load(f)

            except Exception as e:
                raise ValueError(f"Could not load model from {model_name_or_path} neither locally or online: {e}")
        if kwargs.get('scheduler_max_concurrency', None) is None:
            kwargs['scheduler_max_concurrency'] =  self.concurrency

        self.tts_engine = ModelRegistry.get_model_class(
            config['model_type']).from_pretrained(model_name_or_path, **kwargs)

        self.tts_engine.info = ModelRegistry.get_model_info(config['model_type'])

        self._start_orchestrator()

        return self


    async def prepare_for_streaming_generation(self, request: TTSRequest):
        conditioning_config = self.tts_engine.conditioning_config
        if conditioning_config.speaker_embeddings or conditioning_config.gpt_like_decoder_conditioning:
            gpt_cond_latent, speaker_embeddings = await self.tts_engine.get_audio_conditioning(request.speaker_files)
            return partial(self.tts_engine.get_generation_context,
                           gpt_cond_latent=gpt_cond_latent,
                           speaker_embeddings=speaker_embeddings)

    async def generate_speech_async(self, request: TTSRequest) -> Union[AsyncGenerator[TTSOutput, None], TTSOutput]:
        if self._async == False:
            raise RuntimeError("This instance was not created for async generation.")

        self._async = True
        async def process_chunks():
            chunks = []
            try:
                async for chunk in self.orchestrator.run(
                        request=request,
                ):
                    if request.stream:
                        yield chunk
                    chunks.append(chunk)
            except Exception as e:
                self.logger.error(f"Error during speech generation: {e}")
                raise

            if not request.stream:
                yield TTSOutput.combine_outputs(chunks)

        if request.stream:
            return process_chunks()
        else:
            async for result in process_chunks():
                return result

    def generate_speech(self, request: TTSRequest) -> Union[Generator[TTSOutput, None, None], TTSOutput]:
        if self._async == True:
            raise RuntimeError("This instance was created for async generation.")

        self._async = False
        requests = self.split_requests(request)

        if request.stream:
            # Streaming case
            return self._streaming_sync_wrapper(requests)
        else:
            # Non-streaming case
            return self._non_streaming_sync_wrapper(requests)

    async def shutdown(self):
        if self.orchestrator:
            await self.orchestrator.shutdown()
        if self.tts_engine and hasattr(self.tts_engine, 'shutdown'):
            await self.tts_engine.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop())
        self.loop_thread.join()

import asyncio
import json
import logging
import os
import time
import uuid
from typing import AsyncGenerator, Optional, Union, Generator, List, Any

from huggingface_hub import hf_hub_download

from auralis.common.logging.logger import setup_logger, set_vllm_logging_level
from auralis.common.definitions.dto.output import TTSOutput
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.scheduling.three_phase_scheduler import ThreePhaseScheduler
from auralis.models.base import BaseAsyncTTSEngine


class TTS:
    """A high-performance text-to-speech engine optimized for inference speed.

    This class provides an interface for both synchronous and asynchronous speech generation,
    with support for streaming output and parallel processing of multiple requests.
    """

    def __init__(self, scheduler_max_concurrency: int = 10, vllm_logging_level=logging.DEBUG):
        """Initialize the TTS engine.

        Args:
            scheduler_max_concurrency (int): Maximum number of concurrent requests to process.
            vllm_logging_level: Logging level for the VLLM backend.
        """
        set_vllm_logging_level(vllm_logging_level)

        self.scheduler: Optional[ThreePhaseScheduler] = ThreePhaseScheduler(scheduler_max_concurrency)
        self.tts_engine: Optional[BaseAsyncTTSEngine] = None
        self.concurrency = scheduler_max_concurrency
        self.max_vllm_memory: Optional[int] = None
        self.logger = setup_logger(__file__)
        self.loop = None

    def _ensure_event_loop(self):
        """Ensures that an event loop exists and is set."""

        if not self.loop:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        """Load a pretrained model from local path or Hugging Face Hub.
           **THIS METHOD IS SYNCHRONOUS**

        Args:
            model_name_or_path (str): Local path or Hugging Face model identifier.
            **kwargs: Additional arguments passed to the model's from_pretrained method.

        Returns:
            TTS: The TTS instance with loaded model.

        Raises:
            ValueError: If the model cannot be loaded from the specified path.
        """
        from auralis.models.registry import MODEL_REGISTRY

        # Ensure an event loop exists for potential async operations within from_pretrained
        self._ensure_event_loop()

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

        # Run potential async operations within from_pretrained in the event loop
        #async def _load_model():
        #    return MODEL_REGISTRY[config['model_type']].from_pretrained(model_name_or_path, **kwargs)
#
        #self.tts_engine = self.loop.run_until_complete(_load_model()) # to start form the correct loop
        self.tts_engine = MODEL_REGISTRY[config['model_type']].from_pretrained(model_name_or_path, **kwargs)
        return self

    async def _phase_1_prepare_context(self, input_request: TTSRequest):
        """Phase 1: Prepare the generation context (text to tokens, conditioning).
           This happens sequentially.

        Args:
            input_request (TTSRequest): The TTS request to process.

        Returns:
            dict: Dictionary containing parallel inputs and the original request.
        """
        input_request.start_time = time.time()

        parallel_inputs = await self.tts_engine.first_phase(input_request)

        input_request.generators_count = len(parallel_inputs)
        input_request.sequence_buffers = {i: [] for i in range(input_request.generators_count)}
        input_request.completed_generators = 0

        return parallel_inputs

    async def _phase_2_process_tokens(self, *args, **kwargs) -> AsyncGenerator[Any, None]:
        """Phase 2: Process the audio tokens to produce the hidden states.
           This happens in parallel
        Args:
           args: Variable length argument list.
           kwargs: Arbitrary keyword arguments.
        Returns:
            AudioOutputGenerator: Generator yielding audio chunks.
        """
        try:
            item = await self.tts_engine.second_phase(*args, **kwargs)
            return item
        except Exception as e:
            raise e

    async def _phase_3_collect_and_yield(self, *args, **kwargs) -> AsyncGenerator[Any, None]:
        """Third and Final Phase: collect output from the previous phase and yield audio
           This happens in parallel
           Args:
               args: Variable length argument list.
               kwargs: Arbitrary keyword arguments.
           Returns:
                AsyncGenerator: generator yielding audio chunks
        """
        async for item in self.tts_engine.third_phase(*args, **kwargs):
            yield item

    async def _process_request(self, request: TTSRequest):
        """Process a single TTS request through all three phases.

        Args:
            request (TTSRequest): The TTS request to process.

        Yields:
            AsyncGenerator[TTSOutput, None]: Asynchronous generator for audio output.
        """

        try:

            # Phase 1 & 2 & 3: Process audio tokens and generate waveforms in parallel
            async for item in self.scheduler.run(
                    inputs = request,
                    request_id = request.request_id,
                    preprocssing_fn=self.tts_engine.text_preprocessing,
                    first_phase_fn=self._phase_1_prepare_context,
                    second_phase_fn = self._phase_2_process_tokens,
                    third_phase_fn = self._phase_3_collect_and_yield
                ):
                   yield item
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise

    async def generate_speech_async(self, request: TTSRequest) \
            -> Union[AsyncGenerator[TTSOutput, None], TTSOutput]:
        """Generate speech asynchronously from text.

        Args:
            request (TTSRequest): The TTS request to process.

        Returns:
            Union[AsyncGenerator[TTSOutput, None], TTSOutput]: Audio output, either streamed or complete.

        Raises:
            RuntimeError: If instance was not created for async generation.
        """
        self._ensure_event_loop()

        async def process_chunks():
            chunks = []
            try:
                async for chunk in self._process_request(request):
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

    @staticmethod
    def split_requests(request: TTSRequest, max_length: int = 100000) -> List[TTSRequest]:
        """Split a long text request into multiple smaller requests.

        Args:
            request (TTSRequest): The original TTS request.
            max_length (int): Maximum length of text per request.

        Returns:
            List[TTSRequest]: List of split requests.
        """
        if len(request.text) <= max_length:
            return [request]

        text_chunks = [request.text[i:i + max_length]
                       for i in range(0, len(request.text), max_length)]

        return [
            (copy := request.copy(), setattr(copy, 'text', chunk), setattr(copy, 'request_id', uuid.uuid4().hex))[0]
            for chunk in text_chunks
        ]

    async def _process_multiple_requests(self, requests: List[TTSRequest], results: Optional[List] = None) -> Optional[
        TTSOutput]:
        """Process multiple TTS requests in parallel.

        Args:
            requests (List[TTSRequest]): List of requests to process.
            results (Optional[List]): Optional list to store results for streaming.

        Returns:
            Optional[TTSOutput]: Combined audio output if not streaming, None otherwise.
        """
        output_queues = [asyncio.Queue() for _ in requests] if results is not None else None

        async def process_subrequest(idx, sub_request, queue: Optional[asyncio.Queue] = None):
            chunks = []
            async for chunk in self._process_request(sub_request):
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

    def generate_speech(self, request: TTSRequest) -> Union[Generator[TTSOutput, None, None], TTSOutput]:
        self._ensure_event_loop()
        requests = self.split_requests(request)

        if request.stream:
            def streaming_wrapper():
                for sub_request in requests:
                    async def process_stream():
                        try:
                            async for chunk in self._process_request(sub_request):
                                yield chunk
                        except Exception as e:
                            self.logger.error(f"Error during streaming: {e}")
                            raise

                    generator = process_stream()
                    try:
                        while True:
                            # We create a new task per request
                            task = self.loop.create_task(anext(generator))
                            chunk = self.loop.run_until_complete(task) if not self.loop.is_running() else task.result()
                            yield chunk
                    except StopAsyncIteration:
                        pass

            return streaming_wrapper()
        else:
            # Non streaming
            async def _run():
                return await self._process_multiple_requests(requests)

            if self.loop.is_running():
                # If the loop is running, use run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(_run(), self.loop)
                return future.result()
            else:
                # If no loop is running, use run_until_complete
                return self.loop.run_until_complete(_run())

    async def shutdown(self):
        """Shuts down the TTS engine and scheduler."""
        if self.scheduler:
            await self.scheduler.shutdown()
        if self.tts_engine and hasattr(self.tts_engine, 'shutdown'):
            await self.tts_engine.shutdown()
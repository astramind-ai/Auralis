import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Awaitable, Dict, List
from auralis.common.definitions.scheduler import QueuedRequest, TaskState
from auralis.common.logging.logger import setup_logger


class ThreePhaseScheduler:
    """Three-phase asynchronous task scheduler with parallel processing support."""

    def __init__(
            self,
            third_phase_concurrency: int = 10,
            request_timeout: float = None,
            generator_timeout: float = None,
    ):
        """Initialize the scheduler."""
        self.third_phase_concurrency = third_phase_concurrency
        self.request_timeout = request_timeout
        self.generator_timeout = generator_timeout
        self.logger = setup_logger(__file__)

        self.is_running = False
        self.request_queue = None
        self.active_requests: Dict[str, QueuedRequest] = {}
        self.queue_processor_tasks = []
        self.cancel_warning_issued = False

        self.third_phase_sem = None
        self.active_generator_count = 0
        self.generator_count_lock = asyncio.Lock()
        self.cleanup_lock = asyncio.Lock()

    async def start(self):
        """Start the scheduler."""
        if self.is_running:
            return

        self.request_queue = asyncio.Queue()
        self.third_phase_sem = asyncio.Semaphore(self.third_phase_concurrency)
        self.is_running = True
        self.queue_processor_tasks = [
            asyncio.create_task(self._process_queue())
            for _ in range(self.third_phase_concurrency)
        ]

    async def _process_queue(self):
        """Process requests from the queue continuously."""
        while self.is_running:
            try:
                request = await self.request_queue.get()
                if request.state == TaskState.QUEUED:
                    async with self._request_lifecycle(request.id):
                        self.active_requests[request.id] = request
                        await self._process_request(request)
            except asyncio.CancelledError:
                if not self.cancel_warning_issued:
                    self.logger.warning("Queue processing task cancelled")
                    self.cancel_warning_issued = True
                break
            except Exception as e:
                self.logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)

    @asynccontextmanager
    async def _request_lifecycle(self, request_id: str):
        """Manage the lifecycle of a request."""
        try:
            yield
        finally:
            async with self.cleanup_lock:
                self.active_requests.pop(request_id, None)

    async def _process_request(self, request: QueuedRequest):
        """Process a request through all three phases."""
        try:
            self.logger.info(f"Starting request {request.id}")

            # Phase 1: Initial processing - now returns a list of tasks
            first_phase_tasks = await self._handle_first_phase(request)

            # Process Phase 1 outputs concurrently and start Phase 2 and 3 pipelines
            await self._process_phases_concurrently(first_phase_tasks, request)

            if not request.error:
                request.state = TaskState.COMPLETED
                self.logger.info(f"Request {request.id} completed")

        except Exception as e:
            request.error = e
            request.state = TaskState.FAILED
            self.logger.error(f"Request {request.id} failed: {e}")
        finally:
            request.completion_event.set()

    async def _handle_first_phase(self, request: QueuedRequest) -> List[asyncio.Task]:
        """Execute the first phase of request processing and return a list of tasks."""
        request.state = TaskState.PROCESSING_FIRST
        try:
            tasks = [
                asyncio.create_task(request.first_fn(request.input))
            ]  # Ora puoi passare un singolo input a first_fn
            return tasks
        except asyncio.TimeoutError:
            raise TimeoutError(f"First phase timeout after {self.request_timeout}s")

    async def _process_phases_concurrently(
        self, first_phase_tasks: List[asyncio.Task], request: QueuedRequest
    ):
        """Process outputs from the first phase concurrently and start second and third phase pipelines."""
        request.state = TaskState.PROCESSING_SECOND
        request.sequence_buffers = {}
        request.generator_events = {}
        request.generators_count = 0

        # Crea una lista di task per la seconda fase
        second_phase_tasks = []
        for first_task in first_phase_tasks:
            first_results = await asyncio.wait_for(
                first_task, timeout=self.request_timeout
            )
            for first_result in first_results:
                # Avvia la seconda fase per ogni risultato della prima fase
                second_phase_task = asyncio.create_task(
                    self._start_second_phase(request, first_result)
                )
                second_phase_tasks.append(second_phase_task)

            # Attendi il completamento di tutti i task della seconda fase, ma non appena sono pronti
            for second_phase_task in asyncio.as_completed(second_phase_tasks):
                await second_phase_task
        request.state = TaskState.PROCESSING_THIRD

    async def _start_second_phase(self, request: QueuedRequest, first_result: Dict) -> None:
        """Starts the second phase processing for a given first phase result and starts the third phase."""
        try:
            second_phase_result = await asyncio.wait_for(
                request.second_fn(**first_result), timeout=self.request_timeout
            )

            # Assegna un indice univoco al generatore
            generator_idx = request.generators_count
            request.generators_count += 1

            # Inizializza il buffer e l'evento per questo generatore
            request.sequence_buffers[generator_idx] = []
            request.generator_events[generator_idx] = asyncio.Event()

            # Avvia la terza fase (il generatore)
            asyncio.create_task(
                self._process_generator(request, second_phase_result, generator_idx)
            )

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timeout in second phase processing after {self.request_timeout}s"
            )
        except Exception as e:
            self.logger.error(f"Error in second phase processing: {e}")
            if request.error is None:
                request.error = e

    async def _process_generator(
            self, request: QueuedRequest, generator_input: Any, sequence_idx: int
    ):
        """Process a single generator in the third phase."""
        async with self.third_phase_sem:
            try:
                await self._run_generator(request, generator_input, sequence_idx)
            except asyncio.CancelledError:
                self.logger.warning(
                    f"Generator {sequence_idx} cancelled for request {request.id}"
                )
                raise
            except Exception as e:
                self._handle_generator_error(request, sequence_idx, e)
            finally:
                await self._cleanup_generator(request, sequence_idx)

    async def _run_generator(
            self, request: QueuedRequest, generator_input: Any, sequence_idx: int
    ):
        """Run a generator and collect its outputs."""
        generator = request.third_fn(**generator_input)
        buffer = request.sequence_buffers[sequence_idx]
        self.logger.debug(
            f"Starting generator {sequence_idx} for request {request.id} with input {generator_input}"
        )

        try:
            async for item in generator:
                self.logger.debug(f"Generator {sequence_idx} yielded item: {item}")
                event = asyncio.Event()
                buffer.append((item, event))
                event.set()
        except asyncio.TimeoutError as e:
            self.logger.error(
                f"Generator {sequence_idx} timeout for request {request.id}: {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Generator {sequence_idx} exception for request {request.id}: {e}"
            )
            raise
        finally:
            self.logger.debug(
                f"Generator {sequence_idx} completed for request {request.id}"
            )

    def _handle_generator_error(
            self, request: QueuedRequest, sequence_idx: int, error: Exception
    ):
        """Handle errors from a generator."""
        self.logger.error(
            f"Generator {sequence_idx} failed for request {request.id}: {error}"
        )
        if request.error is None:
            request.error = error

    async def _cleanup_generator(self, request: QueuedRequest, sequence_idx: int):
        """Clean up resources after a generator completes."""
        async with self.generator_count_lock:
            self.active_generator_count -= 1
            request.completed_generators += 1
            if sequence_idx in request.generator_events:
                request.generator_events[sequence_idx].set()

    async def _yield_ordered_outputs(
            self, request: QueuedRequest
    ) -> AsyncGenerator[Any, None]:
        """Yield outputs from all generators in strict sequence order."""
        current_index = 0
        last_progress = time.time()

        while not self._is_processing_complete(request):
            if self._check_timeout(last_progress):
                raise TimeoutError("No progress in output generation")

            if request.error:
                raise request.error

            if current_index in request.sequence_buffers:
                buffer = request.sequence_buffers[current_index]
                self.logger.debug(
                    f"Yield check: current_index {current_index}, buffer len: {len(buffer)}"
                )

                if buffer:
                    item, event = buffer.pop(0)
                    try:
                        await asyncio.wait_for(
                            event.wait(), timeout=self.generator_timeout
                        )
                        yield item
                        last_progress = time.time()
                        self.logger.debug(
                            f"Yielded item from sequence {current_index}"
                        )
                        current_index += 1  # Move to the next sequence index
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"Timeout waiting for item in sequence {current_index}"
                        )
                else:
                    await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.01)

    def _is_processing_complete(self, request: QueuedRequest) -> bool:
        """Check if request processing is complete."""
        return (
                request.state in (TaskState.COMPLETED, TaskState.FAILED)
                and request.completed_generators >= request.generators_count
                and all(len(buffer) == 0 for buffer in request.sequence_buffers.values())
        )

    def _check_timeout(self, last_progress: float) -> bool:
        """Check if request has timed out."""
        return (
                self.request_timeout
                and time.time() - last_progress > self.request_timeout
        )

    async def run(
            self,
            inputs: Any,
            first_phase_fn: Callable[[Any], Awaitable[Any]],
            second_phase_fn: Callable[[Any], Awaitable[Any]],
            third_phase_fn: Callable[[Any], AsyncGenerator],
            request_id: str = None,
    ) -> AsyncGenerator[Any, None]:
        """Run a three-phase processing task."""
        if not self.is_running:
            await self.start()

        request = QueuedRequest(
            id=request_id,
            input=inputs,
            first_fn=first_phase_fn,
            second_fn=second_phase_fn,
            third_fn=third_phase_fn,
        )
        await self.request_queue.put(request)

        try:
            async for item in self._yield_ordered_outputs(request):
                yield item
            await asyncio.wait_for(
                request.completion_event.wait(), timeout=self.request_timeout
            )
            if request.error:
                raise request.error

        finally:
            async with self.cleanup_lock:
                self.active_requests.pop(request.id, None)

    async def shutdown(self):
        """Shutdown the scheduler."""
        self.is_running = False

        for task in self.queue_processor_tasks:
            if task and not task.done():
                task.cancel()

        await asyncio.gather(*self.queue_processor_tasks, return_exceptions=True)

        if self.active_requests:
            await asyncio.gather(
                *(
                    request.completion_event.wait()
                    for request in self.active_requests.values()
                ),
                return_exceptions=True,
            )
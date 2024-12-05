import asyncio
import threading
import queue
import inspect
import torch
from auralis import setup_logger
from auralis.common.definitions.types.scheduler import FakeFactoriesForSchedulerProfiling

logger = setup_logger(__file__)

class Profiler:
    @staticmethod
    def profile(fake_factories: FakeFactoriesForSchedulerProfiling, profiling_functions, config):

        async def consume_asyncgen(asyncgen):
            async for _ in asyncgen:
                pass

        async def run_profiling():
            logger.info("Starting Auralis profiling...")

            initial_memory = torch.cuda.memory_allocated()
            initial_memory_gb = initial_memory / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            for function_data, function in zip(fake_factories, profiling_functions):
                if not function_data:
                    continue
                data = function_data(config)

                if inspect.isasyncgenfunction(function):
                    await consume_asyncgen(function(data))
                else:
                    await function(data)

            peak_memory = torch.cuda.max_memory_allocated()
            current_memory = torch.cuda.memory_allocated()

            peak_memory_gb = peak_memory / (1024 ** 3)
            current_memory_gb = current_memory / (1024 ** 3)

            logger.info(
                f"Initial CUDA memory usage: {initial_memory_gb:.2f}GB, "
                f"Peak CUDA memory usage: {peak_memory_gb:.2f}GB, "
                f"Final CUDA memory usage: {current_memory_gb:.2f}GB, "
                f"Memory increase: {(current_memory_gb - initial_memory_gb):.2f} GB"
            )


        # Eseguiamo run_profiling in un thread separato
        result_queue = queue.Queue()

        def worker():
            try:
                result = asyncio.run(run_profiling())
                result_queue.put(result)
            except torch.cuda.OutOfMemoryError:
                logger.error("Profiling failed: CUDA out of memory, try reducing the concurrency")
                result_queue.put(None)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        return result_queue.get()

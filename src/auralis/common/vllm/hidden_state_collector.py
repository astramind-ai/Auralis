import threading
from typing import Optional, Dict, List, Callable
import torch
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from auralis.common.logging.logger import setup_logger


class SyncCollectorWrapper:
    """Wrapper that provides a sync interface for collection while maintaining thread safety"""
    def __init__(self, collector_fn: Callable[[torch.Tensor, str], None], request_id: str):
        self.collector_fn = collector_fn
        self.request_id = request_id

    def __call__(self, hidden_states: Optional[torch.Tensor], request_id: Optional[str] = None):
        """Sync interface for VLLM - uses stored request_id if none provided"""
        self.collector_fn(hidden_states, request_id or self.request_id)

class HiddenStatesCollector:
    def __init__(self):
        self.outputs: Dict[str, List[torch.Tensor]] = {}
        self.collection_ready: Dict[str, threading.Event] = {}
        self.collection_complete: Dict[str, threading.Event] = {}
        self.locks: Dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()
        self.logger = setup_logger(__file__)
        self.states_count: Dict[str, int] = {}
        self.expected_states: Dict[str, int] = {}
        self.notifications: Dict[str, Queue] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    def initialize_request(self, request_id: str):
        """Synchronous initialization for request"""
        with self.global_lock:
            if request_id not in self.locks:
                self.locks[request_id] = threading.Lock()
                self.collection_ready[request_id] = threading.Event()
                self.collection_complete[request_id] = threading.Event()
                self.outputs[request_id] = []
                self.states_count[request_id] = 0
                self.expected_states[request_id] = 1
                self.notifications[request_id] = Queue()
                self.collection_ready[request_id].set()
                self.logger.debug(f"Initialized collector for request {request_id}")

    def sync_collect(self, hidden_states: Optional[torch.Tensor], request_id: str):
        """Synchronous collection method for VLLM callback"""
        if request_id not in self.collection_ready:
            self.logger.error(f"Collector not initialized for request {request_id}")
            # Initialize on demand if needed
            self.initialize_request(request_id)
            return

        try:
            with self.locks[request_id]:
                if hidden_states is not None:
                    self.outputs[request_id].append(hidden_states.clone())
                    self.states_count[request_id] += 1
                    self.logger.debug(f"Collected state {self.states_count[request_id]} for request {request_id}")

                    if self.states_count[request_id] >= self.expected_states[request_id]:
                        self.collection_complete[request_id].set()
                        self.notifications[request_id].put(True)
                else:
                    self.logger.warning(f"Received None hidden states for request {request_id}")
        except Exception as e:
            self.logger.error(f"Error collecting hidden states: {e}")
            raise

    async def get_hidden_states(self, request_id: str, timeout: float = 3.0) -> Optional[torch.Tensor]:
        """Get hidden states for a request with timeout."""
        try:
            if request_id not in self.collection_ready:
                self.logger.error(f"Request {request_id} was never initialized")
                return None

            # Wait for completion using threading.Event
            if not self.collection_complete[request_id].wait(timeout):
                return None

            with self.locks[request_id]:
                outputs = self.outputs.get(request_id, [])
                if not outputs:
                    self.logger.critical(f"No hidden states found for request {request_id}") # most likely due to wrong profiling data dimensions
                    raise ValueError(f"No hidden states found for request {request_id}, "
                                     f"this should not happen, please open an issue on github")

                try:
                    result = torch.cat(outputs, dim=0)
                    self._cleanup_request(request_id)
                    return result
                except Exception as e:
                    self.logger.error(f"Error processing hidden states: {e}")
                    raise

        except Exception as e:
            self.logger.error(f"Error retrieving hidden states: {e}")
            return None

    def _cleanup_request(self, request_id: str):
        """Clean up resources for a request."""
        with self.global_lock:
            self.outputs.pop(request_id, None)
            self.collection_ready.pop(request_id, None)
            self.collection_complete.pop(request_id, None)
            self.locks.pop(request_id, None)
            self.states_count.pop(request_id, None)
            self.expected_states.pop(request_id, None)
            self.notifications.pop(request_id, None)
            self.logger.debug(f"Cleaned up request {request_id}")

    def bind_to_request(self, request_id: str) -> SyncCollectorWrapper:
        """Create a sync wrapper for VLLM callback."""
        # Synchronous initialization
        self.initialize_request(request_id)
        # Pass request_id to wrapper so it's available even if VLLM passes None
        return SyncCollectorWrapper(
            collector_fn=lambda hs, rid: self.sync_collect(hs, rid),
            request_id=request_id
        )
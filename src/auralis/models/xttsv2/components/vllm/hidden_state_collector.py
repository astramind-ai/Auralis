import threading
from typing import Optional, Dict, List, Callable
import torch
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from auralis.common.logging.logger import setup_logger


class SyncCollectorWrapper:
    """Thread-safe wrapper for synchronous hidden state collection.
    
    This wrapper provides a synchronous interface for collecting hidden states
    while maintaining thread safety. It stores a request ID and collection function,
    allowing for simplified collection calls that don't require explicit request IDs.
    """

    def __init__(self, collector_fn: Callable[[torch.Tensor, str], None], request_id: str):
        """Initialize synchronous collector wrapper.

        Args:
            collector_fn (Callable[[torch.Tensor, str], None]): Function to collect
                hidden states with request ID.
            request_id (str): Unique identifier for the collection request.
        """
        self.collector_fn = collector_fn
        self.request_id = request_id

    def __call__(self, hidden_states: Optional[torch.Tensor], request_id: Optional[str] = None):
        """Collect hidden states synchronously.

        Args:
            hidden_states (Optional[torch.Tensor]): Hidden states to collect.
            request_id (Optional[str], optional): Request identifier. If None,
                uses the stored request_id. Defaults to None.
        """
        self.collector_fn(hidden_states, request_id or self.request_id)

class HiddenStatesCollector:
    """Thread-safe collector for model hidden states.
    
    This class manages the collection and retrieval of model hidden states during
    generation, with support for multiple concurrent requests. It provides thread-safe
    operations and timeout-based retrieval.

    The collector maintains separate queues and synchronization primitives for each
    request, allowing for parallel collection of hidden states from different
    generation processes.
    """

    def __init__(self):
        """Initialize hidden states collector.
        
        Sets up thread-safe data structures for collecting and managing hidden states,
        including locks, events, and output storage for multiple requests.
        """
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
        """Initialize collection resources for a new request.

        Sets up all necessary synchronization primitives and storage for a new
        collection request. This method is thread-safe and idempotent.

        Args:
            request_id (str): Unique identifier for the request.
        """
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
        """Synchronously collect hidden states for a request.

        This method is called by VLLM to collect hidden states during generation.
        It handles the thread-safe storage of states and signals completion when
        all expected states are collected.

        Args:
            hidden_states (Optional[torch.Tensor]): Hidden states to collect.
            request_id (str): Request identifier.

        Raises:
            Exception: If there's an error during collection.
        """
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
        """Retrieve collected hidden states for a request.

        This method waits for all hidden states to be collected or until timeout,
        then concatenates and returns the collected states.

        Args:
            request_id (str): Request identifier.
            timeout (float, optional): Maximum time to wait in seconds. Defaults to 3.0.

        Returns:
            Optional[torch.Tensor]: Concatenated hidden states or None if timeout or error.

        Raises:
            ValueError: If no hidden states were collected.
        """
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
        """Clean up resources associated with a completed request.

        This method removes all data structures and synchronization primitives
        associated with a request to prevent memory leaks.

        Args:
            request_id (str): Request identifier to clean up.
        """
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
        """Create a synchronous collector wrapper for a request.

        This method initializes collection resources and returns a wrapper that
        simplifies the collection process for VLLM callbacks.

        Args:
            request_id (str): Request identifier.

        Returns:
            SyncCollectorWrapper: Thread-safe wrapper for collecting hidden states.
        """
        # Synchronous initialization
        self.initialize_request(request_id)
        # Pass request_id to wrapper so it's available even if VLLM passes None
        return SyncCollectorWrapper(
            collector_fn=lambda hs, rid: self.sync_collect(hs, rid),
            request_id=request_id
        )
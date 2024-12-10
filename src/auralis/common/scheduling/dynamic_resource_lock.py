import asyncio
from contextlib import asynccontextmanager

class DynamicResourceLock:
    def __init__(self, max_size):
        """
        This class represents a dynamic resource lock that controls the
        availability of a shared resource. It provides methods to acquire and
        release the resource, as well as the current number of occupied
        resources.

        Args:
            max_size (int): The maximum capacity of the resource lock.
                            If set to a negative value, the resource is considered unlimited.

        Attributes:
            _max_length (int): The maximum capacity.
            _current_occupied (int): The current occupied size of the resource.
            _lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations.
            _condition (asyncio.Condition): A condition variable for managing wait/notify.
            _active_tasks (set): A set to track active tasks.
        """
        self._max_length = max_size
        self._current_occupied = 0
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._active_tasks = set()


    async def acquire(self, item_size):
        """
        Release the resource occupied by the given task ID.

        This method releases the resource previously allocated by the
        `acquire` method. It must be called with the same task ID returned
        by `acquire` to ensure the correct resource is released.

        Args:
            item_size (int): The size of the resource to be acquired.

        Returns:
            task_id (str): The task ID returned by `acquire`.
        """

        async with self._lock:
            if self._max_length < 0: # illimitate resourcea
                return None

            while self._current_occupied + item_size > self._max_length:
                await self._condition.wait()

            self._current_occupied += item_size
            task_id = id(asyncio.current_task())
            self._active_tasks.add((task_id, item_size))
            return task_id

    async def release(self, task_id):
        """
        Release the resource occupied by the given task ID.

        This method releases the resource previously allocated by the
        `acquire` method. It must be called with the same task ID returned
        by `acquire` to ensure the correct resource is released.

        Args:
            task_id (str): The task ID returned by `acquire`.

        Returns:
            None
        """
        async with self._lock:
            task_entry = next((t for t in self._active_tasks if t[0] == task_id), None)
            if task_entry:
                self._active_tasks.remove(task_entry)
                self._current_occupied -= task_entry[1]
                self._condition.notify_all()

    @asynccontextmanager
    async def lock_resource(self, item_size):

        """
        A context manager that ensures only a certain amount of resources are used.

        This context manager is used to ensure that only a certain amount of
        resources are used at any given time. It is used to limit the number of

        Args:
            item_size (int): The size of the resource to be acquired.

        Yields:
            None ( the normal function flow)

        Examples:
            async with dynamic_lock.lock_resource(item_size):
                # Do something
        """
        task_id = await self.acquire(item_size)
        try:
            yield
        finally:
            if task_id is not None:
                await self.release(task_id)

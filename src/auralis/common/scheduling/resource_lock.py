from asyncio import Lock, Condition
from contextlib import asynccontextmanager


class ResourceLock(Lock):
    """
    An asynchronous resource lock that manages concurrent access based on available capacity.
    Extends asyncio.Lock to handle multiple resources with a counting mechanism.

    The lock tracks remaining capacity and allows acquiring/releasing specific amounts of resources,
    ensuring thread-safe access in an async context.

    Attributes:
        _capacity (int): Total capacity of managed resources
        remaining (int): Current number of available resources
        _condition (asyncio.Condition): Condition variable for synchronizing access
    """

    def __init__(self, capacity: int):
        """Initialize the async resource lock with a specified capacity."""
        super().__init__()
        self._capacity = capacity
        self.remaining = capacity
        self._condition = Condition(self)

    async def acquire(self, requested: int = 1) -> bool:
        """
        Asynchronously acquire the specified number of resources if available.

        Args:
            requested (int): Number of resources to acquire. Must be positive and not exceed capacity.

        Returns:
            bool: True if resources were acquired successfully.

        Raises:
            ValueError: If requested amount is invalid or exceeds capacity.
        """
        if not isinstance(requested, int) or requested <= 0:
            raise ValueError("Requested resources must be a positive integer")

        if self._capacity < 0:
            return True  # Some systems might have their internal regulator(like vllm)

        if requested > self._capacity:
            raise ValueError(f"Requested amount ({requested}) exceeds total capacity ({self._capacity})")

        async with self._condition:
            while self.remaining < requested:
                await self._condition.wait()
            self.remaining -= requested
            return True

    async def acquire_nowait(self, requested: int = 1) -> bool:
        """Attempt to acquire resources without waiting."""
        if not isinstance(requested, int) or requested <= 0:
            raise ValueError("Requested resources must be a positive integer")
        if requested > self._capacity:
            raise ValueError(f"Requested amount ({requested}) exceeds total capacity ({self._capacity})")

        async with self._condition:
            if self.remaining < requested:
                return False
            self.remaining -= requested
            return True

    async def release(self, amount: int = 1) -> None:
        """Release the specified number of resources back to the pool."""
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("Release amount must be a positive integer")

        async with self._condition:
            if self.remaining + amount > self._capacity:
                raise ValueError(f"Cannot release {amount} resources - would exceed capacity of {self._capacity}")
            self.remaining += amount
            self._condition.notify_all()

    @property
    def capacity(self) -> int:
        """Get the total capacity of the resource lock."""
        return self._capacity

    @asynccontextmanager
    async def resources(self, amount: int):
        """
        Context manager for acquiring and releasing a specific number of resources.

        Args:
            amount (int): Number of resources to acquire and release

        Yields:
            ResourceLock: The lock instance for use within the context

        Example:
            >>> async with lock.resources(5):
            >>>     # Work with 5 resources
            >>>     pass
        """
        try:
            await self.acquire(amount)
            yield self
        finally:
            await self.release(amount)

    async def __aenter__(self) -> 'ResourceLock':
        """Default async context manager entry - acquires one resource."""
        await self.acquire(1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Default async context manager exit - releases one resource."""
        await self.release(1)
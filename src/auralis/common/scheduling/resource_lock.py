import asyncio
import logging
from asyncio import Lock, Condition
from contextlib import asynccontextmanager

from auralis.common.logging.logger import setup_logger


class ResourceLock:
    """
    An async resource-based lock.
    """

    def __init__(self, capacity: int):

        self._currently_acquired = 0 #used for internal profiling
        self._capacity = capacity
        self.remaining = capacity

        self.logger = setup_logger(__file__)
        # internal lock
        self._lock = Lock()

        # Condition based on internal lock
        self._condition = Condition(self._lock)

    async def acquire(self, requested: int = 1) -> bool:
        """
        Acquisisce 'requested' risorse, se disponibili, altrimenti resta in attesa.
        """
        if not isinstance(requested, int) or requested <= 0:
            raise ValueError("Number of requested resources must be positive")
        if requested > self._capacity:
            raise ValueError(f"Request ({requested}) exceeds maximum capacity ({self._capacity}).")

        async with self._condition:  # lock interno
            while self.remaining < requested:
                await self._condition.wait()
            self.remaining -= requested
            return True

    async def acquire_nowait(self, requested: int = 1) -> bool:
        """
        Try to acquire 'requested' resources without waiting.
        """
        if not isinstance(requested, int) or requested <= 0:
            raise ValueError("You must request at least one resource.")
        if requested > self._capacity:
            raise ValueError(f"Request ({requested}) exceeds maximum capacity ({self._capacity}).")

        async with self._condition:
            if self.remaining < requested:
                return False
            self.remaining -= requested
            return True

    async def release(self, amount: int = 1) -> None:
        """
        Relase 'amount' resources.
        """
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("You must request at least one resource.")

        async with self._condition:

            self.remaining += amount
            self._condition.notify_all()

    @property
    def capacity(self) -> int:
        """Returns the capacity of the resource lock."""
        return self._capacity

    @asynccontextmanager
    async def resources(self, amount: int):
        """
        Context manager to acquire and release resources.
        """
        try:
            await self.acquire(amount)
            self._currently_acquired += 1
            self.logger.debug(f"Currently acquired: {self._currently_acquired}")
            yield self
        finally:
            await self.release(amount)
            self.logger.debug(f"Currently acquired: {self._currently_acquired}")
            self._currently_acquired -= 1

    async def __aenter__(self):
        """acquire 1 resource in the enter."""
        await self.acquire(1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Releases 1 resource in the exit."""
        await self.release(1)

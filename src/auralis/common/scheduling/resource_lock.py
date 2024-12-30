import asyncio
from asyncio import Lock, Condition
from contextlib import asynccontextmanager


class ResourceLock:
    """
    Un lock asincrono che gestisce l’accesso concorrente a N risorse.
    Utilizza internamente un `asyncio.Lock` e un `Condition` per
    sincronizzare i vari acquisiti e rilasci.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("La capacità deve essere un intero positivo.")

        self._capacity = capacity
        self.remaining = capacity

        # Lock interno, diverso da 'self'
        self._lock = Lock()

        # Condition che usa il lock interno
        self._condition = Condition(self._lock)

    async def acquire(self, requested: int = 1) -> bool:
        """
        Acquisisce 'requested' risorse, se disponibili, altrimenti resta in attesa.
        """
        if not isinstance(requested, int) or requested <= 0:
            raise ValueError("Il numero di risorse richieste deve essere un intero positivo.")
        if requested > self._capacity:
            raise ValueError(f"Richiesta ({requested}) eccede la capacità massima ({self._capacity}).")

        async with self._condition:  # lock interno
            while self.remaining < requested:
                await self._condition.wait()
            self.remaining -= requested
            return True

    async def acquire_nowait(self, requested: int = 1) -> bool:
        """
        Tenta di acquisire le risorse richieste senza attendere.
        Restituisce True se riesce, False altrimenti.
        """
        if not isinstance(requested, int) or requested <= 0:
            raise ValueError("Il numero di risorse richieste deve essere un intero positivo.")
        if requested > self._capacity:
            raise ValueError(f"Richiesta ({requested}) eccede la capacità massima ({self._capacity}).")

        async with self._condition:
            if self.remaining < requested:
                return False
            self.remaining -= requested
            return True

    async def release(self, amount: int = 1) -> None:
        """
        Rilascia 'amount' risorse tornando disponibili per altri.
        """
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("Il numero di risorse rilasciate deve essere un intero positivo.")

        async with self._condition:
            if self.remaining + amount > self._capacity:
                raise ValueError(
                    f"Non puoi rilasciare {amount} risorse: supereresti la capacità di {self._capacity}."
                )
            self.remaining += amount
            self._condition.notify_all()

    @property
    def capacity(self) -> int:
        """Ritorna la capacità totale del ResourceLock."""
        return self._capacity

    @asynccontextmanager
    async def resources(self, amount: int):
        """
        Context manager per acquisire e rilasciare automaticamente
        un certo numero di risorse all'entrata e all'uscita dal blocco.
        """
        try:
            await self.acquire(amount)
            yield self
        finally:
            await self.release(amount)

    async def __aenter__(self):
        """Acquisisce di default 1 risorsa in un blocco async with."""
        await self.acquire(1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Rilascia di default 1 risorsa all'uscita dal blocco."""
        await self.release(1)

import asyncio


class AsyncPeekableQueue(asyncio.Queue):
    def peek(self):
        if not self._queue or self.qsize() == 0:
            return None
        return self._queue[0]

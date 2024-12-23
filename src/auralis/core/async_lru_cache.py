import asyncio
from collections import OrderedDict
from functools import wraps
from typing import TypeVar, Callable, ParamSpec, Awaitable, Any, List

P = ParamSpec('P')
R = TypeVar('R')


class AsyncLRUCache:
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lock = asyncio.Lock()

    def __call__(self, func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Create a unique key
            key = hash(str(args) + str(sorted(kwargs.items())))

            async with self.lock:
                # check if the key is in the cache
                if key in self.cache:
                    # Bring it to the end
                    self.cache.move_to_end(key)
                    return self.cache[key]

                # if the cache is full, remove the oldest element
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)

            # if the key is not in the cache, call the function
            result = await func(*args, **kwargs)

            async with self.lock:
                self.cache[key] = result

            return result

        return wrapper
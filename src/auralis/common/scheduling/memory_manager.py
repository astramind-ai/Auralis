# Copyright (c) 2024 Astramind.
# Licensed under the Apache License, Version 2.0.

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import logging
from asyncio import Lock

from auralis import setup_logger
from auralis.common.definitions.scheduler.memory_manager import Block, PagedAllocation

logger = setup_logger(__name__)


class AuralisMemoryManager:
    """
    Buddy-based memory allocator to minimize fragmentation and maintain stable memory usage.
    Supports multiple predefined shapes (2 or 3) defined at initialization.
    """

    def __init__(
        self,
        shapes: List[Tuple[int, int, int]],  # e.g., [(batch, seq, hidden), ...]
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.shapes = shapes

        # Determine max size from given shapes
        self.bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        max_size = 0
        for s in self.shapes:
            b, seq, h = s
            sz = b * seq * h * self.bytes_per_element
            if sz > max_size:
                max_size = sz

        # Round to next power of two
        if max_size == 0:
            max_size = 256
        self.total_size = 2 ** int(np.ceil(np.log2(max_size)))

        # Create a memory pool for the largest shape
        # Actual allocations can be smaller; buddy system splits blocks.
        # Using the largest shape here to ensure enough space.
        # If multiple shapes differ in size, the largest dimension ensures coverage.
        # Pool is just a buffer; shapes are applied at allocation time.
        max_b, max_seq, max_h = max(self.shapes, key=lambda x: x[0]*x[1]*x[2])
        self.memory_pool = torch.zeros(
            (max_b, max_seq, max_h),
            device=self.device,
            dtype=self.dtype
        )
        self.flat_pool = self.memory_pool.view(-1)

        # Buddy system free lists: size -> list of (offset, size)
        self.free_lists = {self.total_size: [(0, self.total_size)]}

        self.allocations: Dict[int, PagedAllocation] = {}
        self._lock = Lock()

        logger.info(
            f"AuralisMemoryManager initialized with total size {self.total_size} bytes"
        )

    async def allocate(self, logical_id: int, shape: Tuple[int, int, int]) -> Optional[List[torch.Tensor]]:
        size = np.prod(shape) * self.bytes_per_element
        if size == 0:
            return [self.flat_pool[:0].view(shape)]
        block_size = 2 ** int(np.ceil(np.log2(size)))

        async with self._lock:
            blk = await self._get_free_block(block_size)
            if blk is None:
                logger.error(f"Allocation failed for shape {shape}")
                return None

            offset, _ = blk
            block_obj = Block(id=offset, size=block_size, shape=shape, is_free=False)
            alloc = PagedAllocation(
                blocks=[block_obj],
                logical_id=logical_id,
                total_size=size,
                shape=shape
            )
            self.allocations[logical_id] = alloc

            start_idx = offset // self.bytes_per_element
            end_idx = start_idx + (size // self.bytes_per_element)
            return [self.flat_pool[start_idx:end_idx].view(shape)]

    async def free(self, logical_id: int):
        async with self._lock:
            allocation = self.allocations.pop(logical_id, None)
            if not allocation:
                return
            for block in allocation.blocks:
                await self._free_block(block.id, block.size)

    async def get_allocation_info(self, logical_id: int) -> Optional[List[torch.Tensor]]:
        async with self._lock:
            alloc = self.allocations.get(logical_id)
            if not alloc:
                return None
            views = []
            esize = self.bytes_per_element
            remaining_size = alloc.total_size
            for block in alloc.blocks:
                start_idx = block.id // esize
                blk_size = min(block.size, remaining_size)
                end_idx = start_idx + (blk_size // esize)
                view = self.flat_pool[start_idx:end_idx].view(alloc.shape)
                views.append(view)
                remaining_size -= blk_size
            return views

    async def get_stats(self) -> Dict:
        async with self._lock:
            free_memory = sum(size * len(blks) for size, blks in self.free_lists.items())
            largest_block = max((size for size, blks in self.free_lists.items() if blks), default=0)
            frag = 0.0 if free_memory == 0 else 1.0 - (largest_block / free_memory)
            return {
                'total_memory': self.total_size,
                'free_memory': free_memory,
                'used_memory': self.total_size - free_memory,
                'active_allocations': len(self.allocations),
                'fragmentation': frag
            }

    async def cleanup(self):
        async with self._lock:
            try:
                del self.memory_pool
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    # Buddy allocator internals
    async def _get_free_block(self, size: int) -> Optional[Tuple[int, int]]:
        if size in self.free_lists and self.free_lists[size]:
            return self.free_lists[size].pop()
        bigger = self._find_bigger_block(size)
        if bigger is None:
            return None
        offset, bsize = self.free_lists[bigger].pop()
        while bsize > size:
            bsize //= 2
            self._add_free_block(offset + bsize, bsize)
        return (offset, bsize)

    async def _free_block(self, offset: int, size: int):
        while True:
            buddy_off = offset ^ size
            fl = self.free_lists.get(size, [])
            buddy_idx = None
            for i, (o, _) in enumerate(fl):
                if o == buddy_off:
                    buddy_idx = i
                    break
            if buddy_idx is not None:
                fl.pop(buddy_idx)
                offset = min(offset, buddy_off)
                size *= 2
            else:
                self._add_free_block(offset, size)
                break

    def _add_free_block(self, offset: int, size: int):
        if size not in self.free_lists:
            self.free_lists[size] = []
        self.free_lists[size].append((offset, size))

    def _find_bigger_block(self, size: int) -> Optional[int]:
        for s in sorted(self.free_lists.keys()):
            if s > size and self.free_lists[s]:
                return s
        return None

    def __del__(self):
        try:
            del self.memory_pool
        except:
            pass

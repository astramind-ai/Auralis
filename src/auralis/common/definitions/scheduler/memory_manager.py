#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Block:
    """Single memory block with linked list capability"""
    id: int
    size: int
    shape: Tuple[int, ...]
    is_free: bool = True
    next_block: Optional['Block'] = None

@dataclass
class PagedAllocation:
    """Logical allocation spanning multiple blocks"""
    blocks: List[Block]
    logical_id: int
    total_size: int
    shape: Tuple[int, ...]

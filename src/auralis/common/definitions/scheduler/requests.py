import time
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from typing import Any, Dict, Optional, List, Callable, TypeVar, AsyncGenerator, Awaitable, Union

import asyncio

from auralis.common.definitions.scheduler.phase_outputs import SecondPhaseOutput, FirstPhaseOutput

T = TypeVar('T')
R = TypeVar('R')


class TaskState(Enum):
    """Enum of states for a task"""
    QUEUED = "QUEUED"
    PROCESSING_FIRST = "PROCESSING_FIRST"
    PROCESSING_SECOND = "PROCESSING_SECOND"
    PROCESSING_THIRD = "PROCESSING_THIRD"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class QueuedRequest:
    """Data structure representing a queued request"""

    def __init__(
            self,
            id: str = None,
            input: Any = None,
            preprocssing_fn: Callable[[Any], Awaitable[Any]] = None,
            first_fn: Callable[[Any], Awaitable[FirstPhaseOutput]] = None,
            second_fn: Callable[[Any], Awaitable[SecondPhaseOutput]] = None,
            third_fn: Callable[[Any], AsyncGenerator] = None,
    ):
        self.id = id
        self.input: Union[FirstPhaseOutput, SecondPhaseOutput] = input
        self.preprocssing_fn = preprocssing_fn
        self.first_fn = first_fn
        self.second_fn = second_fn
        self.third_fn = third_fn
        self.state: TaskState = TaskState.QUEUED
        self.error: Optional[Exception] = None
        self.completion_event = asyncio.Event()
        self.first_phase_result = None
        self.second_phase_result = None
        self.generators_count = 0
        self.completed_generators = 0
        self.sequence_buffers: Dict[int, list] = {}

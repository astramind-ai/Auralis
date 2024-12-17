#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.
from typing import Union, Callable, Coroutine, Any, AsyncGenerator

from auralis.common.definitions.dto.output import TTSOutput
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.definitions.scheduler.contexts import ConditioningContext

BatcherFunction = Callable[
                     [Union[TTSRequest, ConditioningContext]],
                      Coroutine[Any, Any, Union[
                          list[ConditioningContext], ConditioningContext, AsyncGenerator[TTSOutput, None]]]
                 ]
#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.
from typing import Union, Callable, Coroutine, Any, AsyncGenerator

from auralis.common.definitions.dto.output import TTSOutput
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.definitions.scheduler.context import GenerationContext

BatcherFunction = Callable[
                     [Union[TTSRequest, GenerationContext]],
                      Coroutine[Any, Any, Union[
                          list[GenerationContext], GenerationContext, AsyncGenerator[TTSOutput, None]]]
                 ]
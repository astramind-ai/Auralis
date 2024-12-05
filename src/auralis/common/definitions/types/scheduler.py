#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.
from typing import Callable, Optional

Function = Callable
Lambda = Callable

FakeFactoriesForSchedulerProfiling = tuple[Lambda, Optional[Lambda], Lambda]
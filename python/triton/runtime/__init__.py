# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.
from .autotuner import (Autotuner, Config, Heuristics, autotune, heuristics)
from .cache import RedisRemoteCacheBackend, RemoteCacheBackend
from .driver import driver
from .jit import JITFunction, KernelInterface, MockTensor, TensorWrapper, reinterpret
from .errors import OutOfResources, InterpreterError

# Triton 3.4 Inductor assigns launch_*_hook = fn; keep HookChain .add API too.
from .._hookchain_compat import install as _install_hookchain_compat
_install_hookchain_compat()

__all__ = [
    "autotune",
    "Autotuner",
    "Config",
    "driver",
    "Heuristics",
    "heuristics",
    "InterpreterError",
    "JITFunction",
    "KernelInterface",
    "MockTensor",
    "OutOfResources",
    "RedisRemoteCacheBackend",
    "reinterpret",
    "RemoteCacheBackend",
    "TensorWrapper",
]

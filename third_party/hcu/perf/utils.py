#!/usr/bin/env python
# coding=utf-8

import functools
import operator
import importlib
import torch
import triton

from typing import Callable
from packaging.version import Version


# copy from: https://github.com/linkedin/Liger-Kernel/blob/v0.5.5/src/liger_kernel/ops/utils.py#L26
def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def ensure_contiguous(fn):

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):

        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper

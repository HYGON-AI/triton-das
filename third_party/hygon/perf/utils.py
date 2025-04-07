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


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}.")

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def get_glu_block(n):
    BLOCK_SIZE_MAX = 4096
    BLOCK_SIZE = BLOCK_SIZE_MAX if n > (BLOCK_SIZE_MAX * 16) else (triton.next_power_of_2(n) >> 4)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8

    return BLOCK_SIZE, num_warps

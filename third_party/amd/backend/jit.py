from __future__ import annotations, division
import os
import json
import torch
import triton
import logging
import hashlib
import functools
from pathlib import Path
from collections import namedtuple
from typing import Callable, Iterable, Optional, Union
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction as _JITFunction

from triton.runtime.cache import FileCacheManager
from triton.compiler.compiler import CompiledKernel, ASTSource
from triton.runtime.jit import T

from triton._C.libtriton import get_cache_invalidating_env_vars
from triton._utils import find_paths_if, get_iterable_path
from ._utils import triton_version_float, get_cache_dir, get_gpu_label, get_weak_fn_hash, get_triton_label, get_runtime_label

if triton_version_float >= 3.3:
    from triton import knobs

logger = logging.getLogger("triton.fast.jit")


def get_saved_kernel_cache_dir():
    return f"{get_cache_dir()}/saved_kernel"


def get_or_create_metadata(fields):
    if not hasattr(get_or_create_metadata, "_namedtuple_dict"):
        get_or_create_metadata._namedtuple_dict = {}

    key = tuple(fields)
    cls = get_or_create_metadata._namedtuple_dict.get(key)
    if cls is None:
        cls = namedtuple('KernelMetadata', key)
        get_or_create_metadata._namedtuple_dict[key] = cls
    return cls


def get_string_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_saved_kernel_cache_hash(fn):
    device_key = get_gpu_label()
    runtime_key = get_runtime_label()
    triton_key = get_triton_label()
    code_key = get_weak_fn_hash(fn)
    env_vars = get_cache_invalidating_env_vars()
    key = f"{triton_key}-{code_key}-{runtime_key}-{device_key}-{str(sorted(env_vars.items()))}"
    return get_string_hash(key)[:12]


class FastJITFunction(_JITFunction):
    saved_kernel_cache = None
    kernel_cache = {}

    def get_key(self, *args, **kwargs):
        if self.key:
            nargs = dict(zip(self.arg_names, args))
            bound_args = {**nargs, **kwargs}
            if callable(self.key):
                key = self.key(bound_args)
            else:
                key = []
                for k in self.key:
                    if k in bound_args:
                        v = bound_args[k]
                        if hasattr(v, "dtype"):
                            key.append(str(v.dtype))
                        else:
                            key.append(v)
        else:
            _args = [str(v.dtype) if hasattr(v, "dtype") else v for v in args]
            # reorder kwargs name
            _kwargs = []
            for k in sorted(kwargs):
                v = kwargs[k]
                vv = str(v.dtype) if hasattr(v, "dtype") else v
                _kwargs.append(f"{k}={vv}")
            key = _args + _kwargs
        return str(tuple(key))

    def fallback(self, *args, grid, warmup, overwrite=False, **kwargs):
        res = super().run(*args, grid=grid, warmup=warmup, **kwargs)

        # get kernel_path
        asm_files = [Path(p) for c, p in res.metadata_group.items() if not c.endswith(".json")]
        kernel_path = os.path.basename(os.path.dirname(str(asm_files[0])))

        # save signature:path to files in triton cache dir
        key = self.get_key(*args, **kwargs)
        path_cache_dir = f"{get_cache_dir()}/saved_kernel"
        os.makedirs(path_cache_dir, exist_ok=True)
        file_path = f"{path_cache_dir}/{self.saved_cache_key}.json"
        if key not in self.saved_kernel_cache or (key in self.saved_kernel_cache and overwrite):
            self.saved_kernel_cache[key] = kernel_path
            with open(file_path, "w") as f:
                json.dump(self.saved_kernel_cache, f, indent=4)

        return res

    def run(self, *args, grid, warmup, **kwargs):
        if self.saved_kernel_cache is None:
            fpath = f"{get_saved_kernel_cache_dir()}/{self.saved_cache_key}.json"
            if os.path.isfile(fpath):
                try:
                    with open(fpath) as f:
                        self.saved_kernel_cache = json.load(f)
                    logger.info(f"{self.saved_cache_key}: Load saved kernel cache from {fpath}")
                except Exception as e:
                    print(f"{self.saved_cache_key}: Fail to load cache config {fpath} : {e}")
                    self.saved_kernel_cache = {}
            else:
                logger.warning(f"{self.saved_cache_key}: Metadata {fpath} is not exist!")
                self.saved_kernel_cache = {}

        if not self.saved_kernel_cache:
            logger.warning(f"{self.saved_cache_key}: Not found saved kernel in cache, fallback to triton.jit")
            return self.fallback(*args, grid=grid, warmup=warmup, **kwargs)

        bound_args, non_constexpr_vals = {}, []

        for p, v in zip(self.params, args):
            bound_args[p.name] = v
            if not p.is_constexpr:
                non_constexpr_vals.append(v)
        for p in self.params[len(args):]:
            name = p.name
            bound_args[name] = kwargs[name]
            if not p.is_constexpr:
                non_constexpr_vals.append(kwargs[name])

        # get kernel path
        kernel_key = self.get_key(*args, **kwargs)
        if kernel_key not in self.saved_kernel_cache:
            logger.warning(f"{self.saved_cache_key}: Not found saved kernel {kernel_key} in cache, "
                           "fallback to triton.jit")
            return self.fallback(*args, grid=grid, warmup=warmup, **kwargs)
        path = self.saved_kernel_cache[kernel_key]

        # launch saved kernel
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        if path not in self.kernel_cache:
            if triton_version_float >= 3.5: # this code maybe work if >= 3.3
                _, _, _, _, binder = self.device_caches[device]
                # specialization is list[tuple[str, Any]], where first element of tuple is
                # the type and the second parameter is the 'specialization' value.
                _, specialization, _ = binder(*args, **kwargs)

                sigkeys = [x.name for x in self.params]
                sigvals = [x[0] for x in specialization]
                signature = {k: v for (k, v) in zip(sigkeys, sigvals)}

                # constexprs
                constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
                constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}

                src = ASTSource(self, signature, constexprs)

                manager_cls = knobs.cache.manager_class or FileCacheManager
                fn_cache_manager = manager_cls(path)
            else:
                target = driver.active.get_current_target()
                from triton.compiler.compiler import make_backend
                backend = make_backend(target)

                if self.binder is None:
                    self.create_binder(backend)

                _, sig_and_spec, _, _, _ = self.binder(*args, **kwargs)

                bound_vals = tuple(bound_args.values())

                # `None` is nullptr. Implicitly convert to *i8. This needs to be
                # done here rather than when we build the signature as otherwise
                # the kernel cache key could not distinguish between byte pointers
                # and None arguments, resulting in a downstream mismatch:
                sigkeys = [self.params[i].name for i in self.non_constexpr_indices]
                sigvals = sig_and_spec[:len(sigkeys)]
                signature = {k: ('*i8' if (v == 'none') else v) for (k, v) in zip(sigkeys, sigvals)}

                configs = (backend.get_attrs_descriptor(self.params, bound_vals), )
                constant_params = configs[0].get_constants()
                constants = {
                    p.name: v
                    for (v, p) in zip(bound_vals, self.params)
                    if p.is_constexpr or (p.num in constant_params) or v is None
                }
                for i, arg in constants.items():
                    if callable(arg):
                        raise TypeError(f"Callable constexpr at index {i} is not supported")

                # compile the kernel
                src = self.ASTSource(self, signature, constants)#, configs[0])
                fn_cache_manager = FileCacheManager(path)

            metadata_filename = f"{self.__name__[:150]}.json"
            metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
            metadata_path = metadata_group.get(metadata_filename)
            if not metadata_path:
                logger.warning(f"{self.saved_cache_key}: Not found metadata in {get_cache_dir()}/{path}, fallback to triton.jit")
                return self.fallback(*args, grid=grid, warmup=warmup, overwrite=True, **kwargs)
            self.kernel_cache[path] = CompiledKernel(src, metadata_group, None)

        kernel = self.kernel_cache[path]

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                # Arguments are passed as a dict to `grid`, by contract.
                # TODO(jlebar): In the new launch API, pass the compiler flags as a
                # second parameter to `grid`.
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            if hasattr(kernel, "result"):
                kernel = kernel.result()

            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)
            if triton_version_float >= 3.3:
                kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                           knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
            else:
                kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                           kernel.launch_enter_hook, kernel.launch_exit_hook, *non_constexpr_vals)

        return kernel

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None, key=None):
        super().__init__(fn, version=version, do_not_specialize=do_not_specialize,
                        do_not_specialize_on_alignment=do_not_specialize_on_alignment, debug=debug,
                        noinline=noinline, repr=repr, launch_metadata=launch_metadata)
        self.key = key
        cache_hash = get_saved_kernel_cache_hash(self)
        self.saved_cache_key = f"{fn.__name__}-{cache_hash}"[:245] # under 255 char limits


# -----------------------------------------------------------------------------
# `jit` decorator
# -----------------------------------------------------------------------------

def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
    key: Optional[Iterable[str]] = None,
) -> Union[FastJITFunction[T], Callable[[T], FastJITFunction[T]]]:
    """
    Derriving from triton.jit
    """

    def decorator(fn: T) -> FastJITFunction[T]:
        assert callable(fn)
        return FastJITFunction(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
            do_not_specialize_on_alignment=do_not_specialize_on_alignment,
            debug=debug,
            noinline=noinline,
            repr=repr,
            launch_metadata=launch_metadata,
            key=key,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator


triton.jit = jit

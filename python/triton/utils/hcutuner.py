from __future__ import annotations

import os
import uuid
import json
import torch
import uuid
import hashlib
import triton
from collections import defaultdict
from triton.runtime.cache import default_cache_dir, default_dump_dir, default_override_dir

from typing import Dict
from distutils.util import strtobool

from triton.runtime.cache import FileCacheManager, _base32
from triton.compiler.compiler import make_backend, triton_key, ASTSource
from triton.runtime.driver import driver
from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.runtime.jit import mangle_type, DependenciesFinder

file_cache = {}
manager_cache = {}
config_cache = {}


def _get_result_template(arg_names: list, keys:list):
    ret = {
        "arg_names": arg_names,
        "keys": keys,
        "configs": {},
    }
    return ret


def get_config_cache_dir():
    cache_dir = os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()
    return os.path.join(cache_dir, "configs")


class Hcutuner(triton.runtime.Autotuner):
    """
    Re-implements Triton autotune to support custom operations:
    1. save best config to files(under ~/.triton/cache).
    2. cache and restore best config to avoid repetitive tuning
    3. support multi-process tuning
    """
    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None,
                  post_hook=None, prune_configs_by: Dict = None, warmup=None, rep=None,
                  use_cuda_graph=False, do_bench=None, perf_debug=False, perf_profiling=False,
                  always_tuning=False):
        super().__init__(fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, do_bench=do_bench, perf_debug=perf_debug,
                         perf_profiling=perf_profiling)

        # create dir for saving config (e.g. ~/.triton/cache/configs/...)
        device_name = get_gpu_label()
        run_id = str(uuid.uuid4()) # random run_id for multiprocessing
        self.save_config_dir = os.path.join(get_config_cache_dir(),
                                            fn.__name__,
                                            device_name,
                                            run_id)
        os.makedirs(self.save_config_dir, exist_ok=True)

        # Not use the funed configs from cache to save time
        self.flag_restore_tuned_cache = not always_tuning

        self.autotune_param_hash = self._get_param_hash()
        self.autotune_key_hash = get_list_hash(self.keys)
        self.kernel_config_hash = get_config_list_hash(self.configs)

    def run(self, *args, **kwargs):
        if self.flag_restore_tuned_cache:
            self.restore_tuned_cache(*args, **kwargs)
            self.flag_restore_tuned_cache = False

        ret = super().run(*args, **kwargs)
        self.save_config(*args, **kwargs)
        self.cache_config(*args, **kwargs)
        return ret

    def save_config(self, *args, **kwargs):
        """
        Save best config as JSON file to cache dir: triton_cache_dir/configs/...
        """
        fname = os.path.join(self.save_config_dir, "config.json")
        key = get_config_key(self.arg_names, self.keys, *args, **kwargs)
        configs = file_cache[fname] if fname in file_cache else \
                                    _get_result_template(self.arg_names, self.keys)
        if key not in configs['configs']:
            configs['configs'][key] = self.best_config.all_kwargs()
            with open(fname, "w") as f:
                json.dump(configs, f, indent=4)
            file_cache[fname] = configs

    def cache_config(self, *args, **kwargs):
        """
        Cache best config by ConfigCacheManager to avoid repetitive tuning
        """
        if not self.best_config:
            return
        keys = self.keys
        arg_names = self.arg_names
        cache_manager = self.get_config_cache_manager(*args, **kwargs)
        key = cache_manager.key
        cache_json = config_cache[key] if key in config_cache else _get_result_template(arg_names, keys)
        config_key = get_config_key(arg_names, keys, *args, **kwargs)
        if config_key not in cache_json:
            cache_json['configs'][config_key] = self.best_config.all_kwargs()
            # print(f"[hcutuner] added best config to {cache_manager.cache_dir}")
            cache_manager.put(cache_json, "config.json", False)
            config_cache[key] = cache_json

    def restore_tuned_cache(self, *args, **kwargs):
        data = None
        cache_manager = self.get_config_cache_manager(*args, **kwargs)
        data = cache_manager.get("config.json")
        if data and self.keys == data['keys']:
            for k, v in data['configs'].items():
                kt = _create_tuple(k)
                kv = _create_config_args(v)
                self.cache[kt] = triton.Config(**kv)

    def get_config_cache_manager(self, *args, **kwargs):
        key = _base32(_get_cache_hash(self.fn, self.autotune_param_hash, self.autotune_key_hash,
                                      self.kernel_config_hash, *args, **kwargs))
        if key not in manager_cache:
            manager_cache[key] = ConfigCacheManager(key)
        return manager_cache[key]

    def _get_param_hash(self):
        s = f"autotuner params: warmup {self.num_warmups} rep {self.num_reps} use_cuda_graphs {self.use_cuda_graph}"
        # TODO: how to hash the custom hooks?
        #  possible would be str(inspect.Signature().from_callable(self.pre_hook))
        #  maybe not relevant since should not influence the autotuner result
        return get_string_hash(s)


def hcutune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, perf_debug=False, perf_profiling=False,
             always_tuning=False):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function. Its usage is consistent with Triton autotune.
    It's save best config file to `$TRITON_HOME_DIR/.triton/json` directory.

    .. highlight:: python
    .. code-block:: python

        from triton.utils.hcutuner import save_config_gemm
        @triton.utils.hcutune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'], # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    :param always_tuning: Do not use the tuned cache; always perform auto-tuning.
    :type always_tuning: bool
    """

    def decorator(fn):
        return Hcutuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, perf_debug=perf_debug, perf_profiling=perf_profiling,
                         always_tuning=always_tuning)

    return decorator


def get_config_key(arg_names, keys, *args, **kwargs):
    # key format : str(tuple([autotune's key] + [dtypes]))
    nargs = dict(zip(arg_names, args))
    all_args = {**nargs, **kwargs}
    key = [all_args[o] for o in keys if o in all_args]
    dtype = [str(arg.dtype) for _, arg in all_args.items() if hasattr(arg, "dtype")]
    return str(tuple(key + dtype))


def get_gpu_label():
    gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    return gpu_name


class TunedConfig:
    def __init__(self, op_name, device_name, cache):
        self.op_name = op_name
        self.device_name = device_name
        self.cache = cache

    def get_optimal_config(self, *args, **kwargs):
        key = get_config_key(self.cache['arg_names'], self.cache['keys'],
                             *args, **kwargs)
        if key in self.cache['configs']:
            return self.cache['configs'][key]
        return None


class ConfigLoader:
    _instance = None

    def __new__(cls, *args, **kargs):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_dir=None, device=None):
        self.config_dir = get_config_cache_dir() if config_dir is None else config_dir
        self.device = get_gpu_label() if device is None else device
        self.tuned_cache = defaultdict(list)

        if not hasattr(self, "initialized"):
            self.load_all()
            self.initialized = True

    def load_all(self):
        def parse_all_config_files(root_dir):
            res = []
            for fpath in get_config_files(root_dir):
                relative_path = os.path.relpath(fpath, root_dir)
                parts = relative_path.split("/")
                assert len(parts) == 4
                # (filepath, op_name, device_name, run_id)
                res.append((fpath, *parts[:3]))
            return res

        for fpath, op, device, _ in parse_all_config_files(self.config_dir):
            self.tuned_cache[op].append(self.create_tuned_config(fpath, op, device))

    def create_tuned_config(self, fullpath, op_name, device_name):
        try:
            with open(fullpath) as f:
                data = json.load(f)
        except Exception as e:
            raise Exception(f"[hcutuner] Fail to load best config {fullpath} : {e}")
        return TunedConfig(op_name, device_name, data)

    def get_tuned_cache(self, op_name):
        return self.tuned_cache[op_name] if op_name in self.tuned_cache else None


class ConfigCacheManager(FileCacheManager):
    """
    Re-implements Triton's cache manager to:
    1. put/get JSON format file(s)
    2. allow a separate sub-directory for each process
    """
    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = os.getenv("TRITON_DUMP_DIR", "").strip() or default_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = os.getenv("TRITON_OVERRIDE_DIR", "").strip() or default_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()
            if self.cache_dir:
                # random run_id for multiprocessing
                run_id = str(uuid.uuid4())
                self.key_dir = os.path.join(self.cache_dir, self.key)
                self.cache_dir = os.path.join(self.key_dir, run_id)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")

    def get(self, filename):
        data = []
        ret = None
        for fpath in get_config_files(self.key_dir, filename):
            try:
                with open(fpath) as f:
                    data.append(json.load(f))
            except Exception as e:
                raise Exception(f"[hcutuner] Fail to load best config {fpath} : {e}")
        if data:
            ret = _get_result_template(data[0]['arg_names'], data[0]['keys'])
            ret['configs'] = {k: v for d in data for k, v in d['configs'].items()}
        return ret

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = str(uuid.uuid4())
        # we use the PID in case a bunch of these around so we can see what PID made it
        pid = os.getpid()
        # use temp dir to be robust against program interruptions
        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        with open(temp_path, "w") as f:
            json.dump(data, f)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        os.replace(temp_path, filepath)
        os.removedirs(temp_dir)
        return filepath


def _get_src_hash(src: ASTSource):
    """
    Non-key constants are considered not feature for generating kernel.
    So, they are not contain in hash.

    Adapted from: ASTSource.hash()
    """
    def get_fn_hash(fn):
        dependencies_finder = DependenciesFinder(name=fn.__name__, globals={}, src=fn.src)
        dependencies_finder.visit(fn.parse())
        return dependencies_finder.ret

    sorted_sig = [v for k, v in sorted(src.signature.items())]
    key = f"{get_fn_hash(src.fn)}-{src.attrs.hash()}-{sorted_sig}"
    return key


def _get_cache_hash(fn, autotune_param_hash, key_hash, kernel_config_hash, *args, **kwargs):
    """
    Create a hash for locating the best config cache in the triton cache
    directory(~/.triton/cache by default). Where there is a config.json
    containing all the tuned best configs for a specified jit function.

    hash format: triton_code-backend-option-env-src-autotune_params-tune_keys-kernel_configs
    Adapted from: triton/compiler/compiler.py:compile()
    """
    target = driver.active.get_current_target()
    backend = make_backend(target)

    if fn.binder is None:
        fn.create_binder(backend)

    nargs = dict(zip(fn.arg_names, args))
    all_args = {**nargs, **kwargs}
    sigkeys = [fn.params[i].name for i in fn.non_constexpr_indices]
    signature = {k: mangle_type(all_args[k]) for k in sigkeys}
    options = backend.parse_options(kwargs).__dict__

    constants = {}
    for i in fn.constexpr_indices:
        if (name := fn.arg_names[i]) in all_args:
            constants[name] = all_args[name]

    src = fn.ASTSource(fn, signature, constants)
    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))

    env_vars = get_cache_invalidating_env_vars()
    key = f"{triton_key()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
    key += f"-{_get_src_hash(src)}-{autotune_param_hash}-{key_hash}-{kernel_config_hash}"
    return get_string_hash(key)


def _create_tuple(k):
    s = k[1:-1]
    entries = s.split(", ")
    ret = []
    for e in entries:
        if e[0] == "'" or e[0] == '"':
            ret.append(e[1:-1])
        else:
            ret.append(eval(e))
    ret_t = tuple(ret)
    return ret_t


__int_config_args__ = ["num_warps", "num_stages", "num_ctas"]
__int_or_none_config_args__ = ["maxnreg"]
__bool_config_args__ = ["enable_warp_specialization"]
__config_args__ = (
    ["pre_hook"]
    + __int_config_args__
    + __bool_config_args__
    + __int_or_none_config_args__
)
__skip_config_args__ = ["enable_persistent"]


def _create_config_args(args):
    ret = {"kwargs": {}}
    for k, v in args.items():
        if k in __skip_config_args__:
            continue
        if k in __config_args__:
            if k in __int_config_args__:
                ret[k] = int(v)
            elif k in __bool_config_args__:
                ret[k] = bool(strtobool(v))
            elif k in __int_or_none_config_args__:
                try:
                    ret[k] = int(v)
                except ValueError:
                    ret[k] = None
            else:
                ret[k] = v
        else:
            try:
                ret["kwargs"][k] = int(v)
            except ValueError:
                try:
                    ret["kwargs"][k] = bool(strtobool(v))
                except ValueError:
                    print(
                        f"[hcutuner] WARNING: can't determine type of kwarg {k}: {v}"
                    )
                    ret["kwargs"][k] = v
    return ret


def get_config_files(root_dir, filename="config.json"):
    """ Sort by file creation time in ascending order """
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname != filename:
                continue
            files.append(os.path.join(dirpath, filename))
    files.sort(key=lambda f: os.path.getctime(f))
    return files


def get_config_list_hash(configs):
    s = "|"
    for c in configs:
        s += f"{c}|"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_list_hash(l):
    s = "|".join(l)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_string_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

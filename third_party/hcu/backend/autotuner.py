from __future__ import annotations

import os
import builtins
import uuid
import json
import torch
import uuid
import hashlib
import triton
import inspect
from collections import defaultdict

from ._utils import triton_version_float, get_cache_dir, get_dump_dir, get_override_dir, get_weak_fn_hash, get_triton_label, get_gpu_label

from pathlib import Path
from collections import defaultdict, namedtuple

from typing import Dict, Union, Optional, Generic
from distutils.util import strtobool

from triton.runtime import Autotuner
from triton.runtime.cache import FileCacheManager, _base32
# from triton.runtime.jit import T
# from triton.runtime.autotuner import Heuristics
# from triton.utils.jit import patched_init

file_cache = {}
manager_cache = {}
config_cache = {}
kernel_cache = {}

graph_step_cache = {}
graph_cache = {}


def _get_result_template(key: list = []):
    ret = {
        "key": key,
        "configs": {},
        "timings": {},
        "paths": {},
    }
    return ret


def get_config_cache_dir():
    return os.path.join(get_cache_dir(), "configs")


def get_graph_cache_dir():
    return os.path.join(get_cache_dir(), "graph")


def _split_list_balanced(m, n):
    def division_point(i):
        return i * q + min(i, r)

    q, r = divmod(m, n)
    return [list(range(division_point(i), division_point(i+1))) for i in range(n)]


class PruneConfigLoader:
    def __init__(self, configs, world_size=1, rank=0):
        self.configs = configs
        self.indices = _split_list_balanced(len(self.configs), world_size)[rank]

    def __iter__(self):
        for i in self.indices:
            yield self.configs[i]


def get_jit_function(fn):
    while not isinstance(fn, triton.runtime.jit.JITFunction):
        fn = fn.fn
    return fn


def get_config_timings(config_timings, key_config):
    dst = tuple((
        *key_config.all_kwargs().items(),
        key_config.pre_hook,
    ))
    for k, v in config_timings.items():
        src = tuple((
            *k.all_kwargs().items(),
            k.pre_hook,
        ))
        if src == dst:
            return v
    return None


class Hcutuner(Autotuner):
    """
    Re-implements Triton autotune to support custom operations:
    1. save best config to files(under ~/.triton/cache).
    2. cache and restore best config to avoid repetitive tuning
    3. support multi-process tuning
    """
    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None, post_hook=None,
                 prune_configs_by: Optional[Dict] = None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None,
                 cache_results=False, perf_debug=False, perf_profiling=False):
        if triton_version_float > 3.3:
            super().__init__(fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                            post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                            use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results)
        else:
            super().__init__(fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                            post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                            use_cuda_graph=use_cuda_graph, do_bench=do_bench, perf_debug=perf_debug,
                            perf_profiling=perf_profiling)

        # create dir for saving config (e.g. ~/.triton/cache/configs/...)
        device_name = get_gpu_label()
        run_id = str(uuid.uuid4()) # random run_id for multiprocessing
        self.jit_fn = get_jit_function(fn)
        self._fn_name = self.jit_fn.__name__
        self.save_config_dir = os.path.join(get_config_cache_dir(),
                                            self._fn_name,
                                            device_name,
                                            run_id)
        os.makedirs(self.save_config_dir, exist_ok=True)

        self.param_hash = self._get_param_hash()
        self.key_hash = get_list_hash(self.keys)
        self.configs_hash = get_config_list_hash(self.configs)

        # dict(<key: best config's timings>)
        self._configs_timings = {}

        # resotre the configs
        if not os.getenv("TRITON_HCUTUNE_ALWAYS_TUNING", "0") == "1":
            self.restore_tuned_cache()

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        if isinstance(self.fn, triton.runtime.autotuner.Heuristics):
            for v, heur in self.fn.values.items():
                kwargs[v] = heur({**dict(zip(self.fn.arg_names, args)), **kwargs})
            fn = self.fn.fn
        else:
            fn = self.fn
        if os.getenv("TRITON_HCUTUNE_CONFIGS_SHARDING", "0") == "1":
            rank = eval(os.getenv("TRITON_HCUTUNE_LOCAL_RANK", "0").strip())
            world_size = eval(os.getenv("TRITON_HCUTUNE_WORLD_SIZE", "1").strip())
            config_loader = PruneConfigLoader(self.prune_configs(kwargs), world_size, rank)
        else:
            config_loader = self.prune_configs(kwargs)
        for config in config_loader:
            if config:
                print(f"Warmup kernel {fn.__name__} with config {config.all_kwargs()}")
                ret.append(fn.warmup(
                    *args,
                    **kwargs,
                    **config.all_kwargs(),
                ))
        self.nargs = None
        return ret

    def run(self, *args, **kwargs):
        if os.getenv("TRITON_HCUTUNE_COMPILE_ONLY") == "1":
            del kwargs['warmup']
            return self.warmup(*args, **kwargs)

        ret = super().run(*args, **kwargs)

        # got new config
        if (hasattr(self, 'configs_timings') and self.configs_timings) or len(self.configs) == 1:
            _, key = get_config_key(self.arg_names, self.keys, *args, **kwargs)

            if hasattr(self, 'configs_timings'):
                self._configs_timings[key] = get_config_timings(self.configs_timings, self.best_config)
            else:
                self._configs_timings[key] = [0., 0., 0.]

            self.save_config(ret, *args, **kwargs)
            self.cache_config(*args, **kwargs)

            if os.getenv("TRITON_HCUTUNE_GRAPH_TRACE", "0") == "1":
                self.save_graph_config(ret, *args, **kwargs)

            self.configs_timings = None

        return ret

    def save_config(self, compiled, *args, **kwargs):
        """
        Save best config as JSON file to cache dir: triton_cache_dir/configs/...
        """
        fname = os.path.join(self.save_config_dir, "config.json")
        key_name, key = get_config_key(self.arg_names, self.keys, *args, **kwargs)
        _key = str(key)
        configs = file_cache[fname] if fname in file_cache else _get_result_template()
        if _key not in configs['configs']:
            configs['key'] = key_name
            configs['configs'][_key] = self.best_config.all_kwargs()
            configs['timings'][_key] = self._configs_timings[key]
            if triton_version_float >= 3.5:
                asm_files = [Path(p) for c, p in compiled.metadata_group.items() if not c.endswith(".json")]
                kernel_path = os.path.dirname(str(asm_files[0]))
            else:
                kernel_path = os.path.basename(compiled.perf_ir_path)
            configs['paths'][_key] = kernel_path
            with open(fname, "w") as f:
                json.dump(configs, f, indent=4)
            file_cache[fname] = configs

    def save_graph_config(self, compiled, *args, **kwargs):
        """
        Save best config as JSON file to cache dir: triton_cache_dir/graph/...
        """
        key_name, key = get_config_key(self.arg_names, self.keys, *args, **kwargs)
        node_key = f"{self._fn_name}-{str(key)}-{str(uuid.uuid4()).replace('-', '_')}"
        if node_key not in graph_step_cache:
            graph_step_cache[node_key] = {
                'key': key_name,
                'config': self.best_config.all_kwargs(),
                'timings': self._configs_timings[key],
                'path': os.path.basename(compiled.perf_ir_path),
            }

    def cache_config(self, *args, **kwargs):
        """
        Cache best config by ConfigCacheManager to avoid repetitive tuning
        """
        if not self.best_config:
            return
        keys = self.keys
        arg_names = self.arg_names
        cache_manager = self.get_config_cache_manager()
        key = cache_manager.key
        config_key_name, config_key = get_config_key(arg_names, keys, *args, **kwargs)
        _config_key = str(config_key)
        cache_json = config_cache[key] if key in config_cache else _get_result_template()
        if _config_key not in cache_json:
            cache_json['key'] = config_key_name
            cache_json['configs'][_config_key] = self.best_config.all_kwargs()
            cache_json['timings'][_config_key] = self._configs_timings[config_key]
            # print(f"[hcutuner] added best config to {cache_manager.cache_dir}")
            cache_manager.put(cache_json, "config.json", False)
            config_cache[key] = cache_json

    def restore_tuned_cache(self):
        cache_manager = self.get_config_cache_manager()
        data = cache_manager.get("config.json")
        if data:
            for k, v in data['configs'].items():
                kt = _create_tuple(k)
                kv = _create_config_args(v)
                config = triton.Config(**kv)
                self.cache[kt] = config
                self._configs_timings[kt] = data['timings'][str(kt)]

    def get_config_cache_manager(self):
        key = _base32(_get_cache_hash(self.jit_fn, self.param_hash,
                                      self.key_hash, self.configs_hash))
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
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False, perf_debug=False, perf_profiling=False):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
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
        'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
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
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    "type cache_results: bool
    """

    def decorator(fn):
        return Hcutuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results,
                         perf_debug=perf_debug, perf_profiling=perf_profiling)

    return decorator


def get_config_key(arg_names, keys, *args, **kwargs):
    # key format : str(tuple([autotune's key] + [dtypes]))
    nargs = dict(zip(arg_names, args))
    all_args = {**nargs, **kwargs}

    k_name, k_val = [], []
    for k in keys:
        if k in all_args:
            k_name.append(k)
            k_val.append(all_args[k])

    other, dtype = [], []
    for n, arg in all_args.items():
        if hasattr(arg, "dtype"):
            dtype.append(str(arg.dtype))
            other.append(n)

    return k_name + other, tuple(k_val + dtype)


def merge_caches(data):
    """ merge a list of config cache """
    res = _get_result_template(max([d['key'] for d in data], key=len))
    _configs, _timings, _paths = defaultdict(list), defaultdict(list), defaultdict(list)
    for d in data:
        for k, v in d['configs'].items():
            _configs[k].append(v)
        for k, v in d['timings'].items():
            _timings[k].append(v)
        for k, v in d['paths'].items():
            _paths[k].append(v)
    assert len(_configs) == len(_timings)
    if _paths:
        assert len(_paths) == len(_configs)
    # fill with the best config
    for k, v in _timings.items():
        min_v = builtins.min(v)
        i = v.index(min_v)
        res['timings'][k] = v[i]
        res['configs'][k] = _configs[k][i]
        if _paths:
            res['paths'][k] = _paths[k][i]
    return res


class TunedConfig:
    def __init__(self, op_name, device_name, cache):
        self.op_name = op_name
        self.device_name = device_name
        self.cache = cache

    def get_optimal_config(self, key: Union[list, dict, tuple, str]):
        def handle(value):
            if hasattr(value, "dtype"): # torch tensor
                return str(value.dtype)
            elif isinstance(value, torch.dtype):
                return str(value)
            else: # int, float, bool, str
                return value

        if not isinstance(key, str):
            keys = self.cache['key']
            if isinstance(key, (list, tuple)):
                key = dict(zip(keys, key))
            _key = [handle(key[n]) for n in keys]
            key = str(_key[0] if len(_key) == 1 else tuple(_key))

        if key in self.cache['configs']:
            return self.cache['configs'][key]
        return None


class ConfigLoader:
    def __init__(self, config_dir=None, device=None):
        self.config_dir = get_config_cache_dir() if config_dir is None else config_dir
        self.device = get_gpu_label() if device is None else device
        self.tuned_cache = {}
        self.kernel_device_map = defaultdict(list)
        self.load_all()

    def load_all(self):
        def parse_all_config_files(root_dir):
            res = []
            for fpath in get_config_files(root_dir):
                relative_path = os.path.relpath(fpath, root_dir)
                parts = relative_path.split("/")
                assert len(parts) == 4
                # (filepath, op_name, device_name, run_id)
                res.append((fpath, *parts[:3]))
                if parts[1] not in self.kernel_device_map[parts[0]]:
                    self.kernel_device_map[parts[0]].append(parts[1])
            return res

        tuned_cache = defaultdict(list)
        for fpath, op, device, _ in parse_all_config_files(self.config_dir):
            try:
                with open(fpath) as f:
                    tuned_cache[(op, device)].append(json.load(f))
            except Exception as e:
                raise Exception(f"[hcutuner] Fail to load best config {fpath} : {e}")

        for k, v in tuned_cache.items():
            cache = merge_caches(v)
            self.tuned_cache[k] = TunedConfig(k[0], k[1], cache)

    def get_tuned_cache(self, op_name, device_name):
        key = (op_name, device_name)
        return self.tuned_cache[key] if key in self.tuned_cache else None


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
            self.cache_dir = get_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = get_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = get_cache_dir()
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
            ret = merge_caches(data)
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


def _get_cache_hash(fn, param_hash, key_hash, configs_hash):
    """
    Create a hash for locating the best config cache in the triton cache
    directory(~/.triton/cache by default). Where there is a config.json
    containing all the tuned best configs for a specified jit function.

    hash format: "triton_label-function-autotune_params-key-configs"
    Adapted from: triton/compiler/compiler.py:compile()
    """
    key = f"{get_triton_label()}-{get_weak_fn_hash(fn)}-{param_hash}-{key_hash}-{configs_hash}"
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
                    ret["kwargs"][k] = v
    return ret


def get_config_files(root_dir, filename="config.json"):
    """ Sort by file creation time in ascending order """
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            # In a multi-process interrupted-and-resume tuning scenario, the following issue may occur:
            # The cache manager uses a temporary directory `tmp.pid_xxx` to assist in writing config.json to the cache directory,
            # and this temporary directory will be deleted once the cache writing is completed.
            # At that moment, if another process happens to walk the temporary directory while collecting config.json,
            # the file may no longer exist when it later tries to access it, leading to a operation error.
            if fname != filename or '/tmp.pid_' in dirpath:
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


#class KernelWithConfigInterface(Generic[T]):
#    """
#    adptived from KernelInterface to run with get_config=True
#    """
#    run: T
#
#    def __getitem__(self, grid) -> T:
#        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, get_config=True,
#                                                *args, **kwargs)


#class JITFunctionWithConfig(KernelWithConfigInterface[T]):
#    """
#    adptived from JITFunction to get optimal config if get_config=True
#    """
#    def __init__(self, fn, get_config_fn=None, get_key_fn=None):
#        self.fn = fn
#        self.__name__ = fn.__name__
#        self.arg_names = fn.arg_names
#        self.get_config_fn = get_config_fn
#        self.get_key_fn = get_key_fn
#
#    def run(self, *args, **kwargs):
#        if 'get_config' in kwargs and kwargs['get_config']:
#            config = self.get_config(*args, **kwargs)
#            if config:
#                kwargs = {**kwargs, **config}
#            del kwargs['get_config']
#        return self.fn.run(*args, **kwargs)
#
#    def warmup(self, *args, **kwargs):
#        if 'get_config' in kwargs and kwargs['get_config']:
#            config = self.get_config(*args, **kwargs)
#            if config:
#                kwargs = {**kwargs, **config}
#            del kwargs['get_config']
#        return self.fn.warmup(*args, **kwargs)
#
#    def get_config(self, *args, **kwargs):
#        config = None
#        nargs = {**dict(zip(self.arg_names, args)), **kwargs}
#        if self.get_config_fn:
#            config = self.get_config_fn(nargs)
#        else:
#            # load config from cache
#            tuned = triton.utils.global_config_loader.get_tuned_cache(self.__name__, get_gpu_label())
#            if tuned:
#                if self.get_key_fn:
#                    key = self.get_key_fn(tuned.cache['configs'], nargs)
#                else:
#                    key = [nargs[k] for k in tuned.cache['key']]
#                config = tuned.get_optimal_config(key)
#                if not config:
#                    print(
#                        f"[hcutune_configured] WARNING: Not found optimal config for {self.__name__}, "
#                        f"config key: {key} !!!"
#                    )
#
#        return config


#def hcutune_configured(get_config_fn=None, get_key_fn=None):
#    """
#    Decorator for run a :code:`triton.jit`'d function with tuned optimal config.
#
#    .. highlight:: python
#    .. code-block:: python
#
#        @triton.utils.hcutune_configured
#        @triton.jit
#        ...
#
#    :param get_config_fn: a user-provided function to get or create optimal config
#    :type get_config_fn: lambda META
#    :param get_key_fn: a user-provided function to create a key to load optimal config from cache
#    :type get_key_fn: lambda config_cache, META
#    """
#
#    def decorator(fn):
#        return JITFunctionWithConfig(fn, get_config_fn, get_key_fn)
#
#    return decorator


# class KernelCacheManager(FileCacheManager):
#     """
#     Re-implements Triton's cache manager to:
#       Make group data to fit cache dir.

#     Triton writes kernel binary and metadata abs paths to the group file, and these hard-coded
#     paths become invalid when the Triton cache dir differs from the cache generation.
#     """
#     def get_group(self, filename: str) -> Optional[Dict[str, str]]:
#         grp_filename = f"__grp__{filename}"
#         if not self.has_file(grp_filename):
#             return None
#         grp_filepath = self._make_path(grp_filename)
#         with open(grp_filepath) as f:
#             grp_data = json.load(f)
#         child_paths = grp_data.get("child_paths", None)
#         # Invalid group data.
#         if child_paths is None:
#             return None
#         result = {}
#         for c, p in child_paths.items():
#             p = os.path.join(self.cache_dir, os.path.basename(p))
#             if os.path.exists(p):
#                 result[c] = p
#         return result


# def run_saved_kernel(fn, path, *args, grid, **kwargs):
#     bound_args, non_constexpr_vals = {}, []

#     if isinstance(fn, Heuristics):
#         for v, heur in fn.values.items():
#             kwargs[v] = heur({**dict(zip(fn.arg_names, args)), **kwargs})
#         fn = fn.fn

#     for p, v in zip(fn.params, args):
#         bound_args[p.name] = v
#         if not p.is_constexpr:
#             non_constexpr_vals.append(v)
#     for p in fn.params[len(args):]:
#         name = p.name
#         bound_args[name] = kwargs[name]
#         if not p.is_constexpr:
#             non_constexpr_vals.append(kwargs[name])

#     device = driver.active.get_current_device()
#     stream = driver.active.get_current_stream(device)

#     if path not in kernel_cache:
#         target = driver.active.get_current_target()
#         from ..compiler.compiler import make_backend
#         backend = make_backend(target)

#         if fn.binder is None:
#             fn.create_binder(backend)
#         fn.CompiledKernel.__init__ = patched_init

#         _, sig_and_spec, _, _, _ = fn.binder(*args, **kwargs)

#         bound_vals = tuple(bound_args.values())

#         # `None` is nullptr. Implicitly convert to *i8. This needs to be
#         # done here rather than when we build the signature as otherwise
#         # the kernel cache key could not distinguish between byte pointers
#         # and None arguments, resulting in a downstream mismatch:
#         sigkeys = [fn.params[i].name for i in fn.non_constexpr_indices]
#         sigvals = sig_and_spec[:len(sigkeys)]
#         signature = {k: ('*i8' if (v == 'none') else v) for (k, v) in zip(sigkeys, sigvals)}

#         configs = (backend.get_attrs_descriptor(fn.params, bound_vals), )
#         constant_params = configs[0].get_constants()
#         constants = {
#             p.name: v
#             for (v, p) in zip(bound_vals, fn.params)
#             if p.is_constexpr or (p.num in constant_params) or v is None
#         }
#         for i, arg in constants.items():
#             if callable(arg):
#                 raise TypeError(f"Callable constexpr at index {i} is not supported")

#         # compile the kernel
#         src = fn.ASTSource(fn, signature, constants)#, configs[0])

#         fn_cache_manager = KernelCacheManager(path)

#         metadata_filename = f"{fn.__name__[:150]}.json"
#         metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
#         metadata_path = metadata_group.get(metadata_filename)
#         assert metadata_path is not None
#         kernel_cache[path] = fn.CompiledKernel(src, metadata_group, None)

#     kernel = kernel_cache[path]

#     # canonicalize grid
#     assert grid is not None
#     if callable(grid):
#         # Arguments are passed as a dict to `grid`, by contract.
#         # TODO(jlebar): In the new launch API, pass the compiler flags as a
#         # second parameter to `grid`.
#         grid = grid(bound_args)
#     grid_size = len(grid)
#     grid_0 = grid[0]
#     grid_1 = grid[1] if grid_size > 1 else 1
#     grid_2 = grid[2] if grid_size > 2 else 1

#     # launch kernel
#     launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)
#     kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
#                 fn.CompiledKernel.launch_enter_hook, fn.CompiledKernel.launch_exit_hook, *non_constexpr_vals)

#     return kernel


class Graphtuner:
    def __init__(self, fn):
        self.fn = fn
        device_name = get_gpu_label()
        run_id = str(uuid.uuid4()) # random run_id for multiprocessing
        self.save_graph_dir = os.path.join(get_graph_cache_dir(),
                                           fn.__name__,
                                           device_name,
                                           run_id)
        os.makedirs(self.save_graph_dir, exist_ok=True)
        self.arg_names = inspect.signature(self.fn).parameters.keys()
        self.traced = set()

    def __call__(self, *args, **kwargs):
        nargs = {k: f"{str(v.shape)}-{str(v.dtype)}"
                 if hasattr(v, 'dtype') else v for k, v in zip(self.arg_names, args)}
        bound_args = {**nargs, **kwargs}
        hash = get_string_hash(str(bound_args))
        env_graph_trace = os.getenv("TRITON_HCUTUNE_GRAPH_TRACE", "0")
        env_hcutune = os.getenv("TRITON_HCUTUNE", "0")

        if hash not in self.traced:
            # Save the graph only on the first execution to prevent function
            # from repeatedly saving the graph because of the benchmark
            os.environ['TRITON_HCUTUNE_GRAPH_TRACE'] = '1'
            os.environ['TRITON_HCUTUNE'] = '1'
            self.traced.add(hash)

        res = self.fn(*args, **kwargs)
        self.save_graph_config()
        # restore env
        os.environ['TRITON_HCUTUNE_GRAPH_TRACE'] = env_graph_trace
        os.environ['TRITON_HCUTUNE'] = env_hcutune
        return res

    def save_graph_config(self):
        if graph_step_cache:
            fname = os.path.join(self.save_graph_dir, "config.json")
            key = self.do_hash(graph_step_cache)
            if key not in graph_cache:
                graph_cache[key] = graph_step_cache.copy()
                with open(fname, "w") as f:
                    json.dump(graph_cache, f, indent=4)
            graph_step_cache.clear()
    
    def do_hash(self, data):
        s = json.dumps(data)
        return get_string_hash(s)


def graphtune(fn):
    return Graphtuner(fn)


class GraphConfigLoader:
    def __init__(self, config_dir=None, device=None):
        self.config_dir = get_graph_cache_dir() if config_dir is None else config_dir
        self.device = get_gpu_label() if device is None else device
        self.cache = defaultdict(list)
        self.graph_device_map = defaultdict(list)
        self.load_all()

    def load_all(self):
        def parse_all_config_files(root_dir):
            res = []
            for fpath in get_config_files(root_dir):
                relative_path = os.path.relpath(fpath, root_dir)
                parts = relative_path.split("/")
                assert len(parts) == 4
                # (filepath, graph_name, device_name, run_id)
                res.append((fpath, *parts[:3]))
                if parts[1] not in self.graph_device_map[parts[0]]:
                    self.graph_device_map[parts[0]].append(parts[1])
            return res

        for fpath, graph, device, _ in parse_all_config_files(self.config_dir):
            try:
                with open(fpath) as f:
                    _data = json.load(f)
                if len(_data) == 1 and not next(iter(_data.values())):
                    continue
                _key = (graph, device)
                _cache = self.cache[_key]
                for v in _data.values():
                    _cache.append(v)
            except Exception as e:
                raise Exception(f"[hcutuner] Fail to load graph config {fpath} : {e}")


triton.autotune = hcutune
triton.runtime.autotune = hcutune

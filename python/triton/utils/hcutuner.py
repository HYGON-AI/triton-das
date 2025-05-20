from __future__ import annotations

import os
import json
import torch
import triton
from triton.runtime.cache import default_cache_dir

file_cache = {}


def _get_result_template(arg_names: list, keys:list):
    ret = {
        "arg_names": arg_names,
        "keys": keys,
        "configs": {},
    }
    return ret


def get_config_cache_dir():
    return os.path.join(default_cache_dir(), "configs")


class Hcutuner(triton.runtime.Autotuner):
    """
    Re-implements Triton autotune to support custom operations:
    1. save best config to files(under ~/.triton/json).
    """

    def run(self, *args, **kwargs):
        ret = super().run(*args, **kwargs)
        self.save_config(self.best_config, *args, **kwargs)
        return ret

    def save_config(self, config, *args, **kwargs):
        if not hasattr(self, 'save_config_dir') or self.save_config_dir is None:
            device_name = get_gpu_label()
            self.save_config_dir = os.path.join(get_config_cache_dir(),
                                                self.fn.__name__,
                                                device_name)
            os.makedirs(self.save_config_dir, exist_ok=True)
        fname = os.path.join(self.save_config_dir, "config.json")
        key = get_config_key(self.arg_names, self.keys, *args, **kwargs)
        configs = file_cache[fname] if fname in file_cache else \
                                    _get_result_template(self.arg_names, self.keys)
        if key not in configs['configs']:
            configs['configs'][key] = config.all_kwargs()
            with open(fname, "w") as f:
                json.dump(configs, f, indent=4)
            file_cache[fname] = configs


def hcutune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, perf_debug=False, perf_profiling=False):
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
    """

    def decorator(fn):
        return Hcutuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, perf_debug=perf_debug, perf_profiling=perf_profiling)

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
        self.tuned_cache = {}

        if not hasattr(self, "initialized"):
            self.load_all()
            self.initialized = True

    def load_all(self):
        def parse_all_config_files(root_dir):
            res = []
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename != "config.json":
                        continue
                    fullpath = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(fullpath, root_dir)
                    parts = relative_path.split("/")
                    assert len(parts) == 3
                    # (filepath, op_name, device_name)
                    res.append((fullpath, parts[0], parts[1]))
            return res

        for fpath, op, device in parse_all_config_files(self.config_dir):
            self.tuned_cache[op] = self.create_tuned_config(fpath, op, device)

    def create_tuned_config(self, fullpath, op_name, device_name):
        try:
            with open(fullpath) as f:
                data = json.load(f)
        except Exception as e:
            raise Exception(f"[hcutuner] Fail to load best config {fullpath} : {e}")
        return TunedConfig(op_name, device_name, data)

    def get_tuned_cache(self, op_name):
        return self.tuned_cache[op_name] if op_name in self.tuned_cache else None

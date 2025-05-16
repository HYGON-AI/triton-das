from __future__ import annotations

import os
import json
import torch
import triton
from triton.runtime.cache import get_home_dir
from typing import Dict

config_cache = {}


class Hcutuner(triton.runtime.Autotuner):
    """
    Re-implements Triton autotune to support custom operations, e.g. save best config to file.
    """

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None,
                 post_hook=None, prune_configs_by: Dict = None, warmup=None, rep=None,
                 use_cuda_graph=False, do_bench=None, perf_debug=False, perf_profiling=False,
                 do_save_config=None, save_config_dir=None):
        if do_save_config:
            self.do_save_config = do_save_config
        else:
            self.do_save_config = default_save_config

        # make output config dir, like ~/.triton/json/function_name/configs
        if not save_config_dir:
            save_config_dir = os.path.join(get_home_dir(), ".triton", "json")

        self.save_config_dir = os.path.join(save_config_dir, fn.__name__, "configs")
        os.makedirs(self.save_config_dir, exist_ok=True)

        super().__init__(fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                    post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                    use_cuda_graph=use_cuda_graph, do_bench=do_bench, perf_debug=perf_debug,
                    perf_profiling=perf_profiling)

    def run(self, *args, **kwargs):
        ret = super().run(*args, **kwargs)
        if self.do_save_config:
            nargs = dict(zip(self.arg_names, args))
            all_args = {**nargs, **kwargs}
            self.do_save_config(self.save_config_dir, self.best_config, self.keys, **all_args)
        return ret


def hcutune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, perf_debug=False, perf_profiling=False,
             do_save_config=None, save_config_dir=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function. Its usage is consistent with Triton autotune.
    `do_save_config` has been added to specify a function for saving the best config. `save_config_dir` has
    been added to specify top output dir for configs, if it's None, set path to `<TRITON_HOME_DIR>/.triton/json`.

    .. highlight:: python
    .. code-block:: python

        from triton.utils.hcutuner import save_config_gemm
        @triton.utils.hcutune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'], # the two above configs will be evaluated anytime
                         # the value of x_size changes
          do_save_config=save_config_gemm,
          save_config_dir=...
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
    :param do_save_config: a function to save best config
    :type do_save_config: lambda fn: JITFunction, keys: list, best_config: triton.Config, *args, **kwargs
    """

    def decorator(fn):
        return Hcutuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, perf_debug=perf_debug, perf_profiling=perf_profiling,
                         do_save_config=do_save_config, save_config_dir=save_config_dir)

    return decorator


def get_dtypes(nargs):
    return '_'.join([str(arg.dtype).replace('torch.', '')
                     for _, arg in nargs.items() if hasattr(arg, "dtype")])


def get_gpu_label():
    gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    return gpu_name


def _save_config(dirname, config, keys, create_filename_and_key, **kwargs):
    fn, key = create_filename_and_key(kwargs, keys)
    key = str(key)
    fname = os.path.join(dirname, fn)

    configs = config_cache[fname] if fname in config_cache else {}
    if key not in configs:
        configs[key] = config.all_kwargs()
        with open(fname, "w") as f:
            json.dump(configs, f)
        config_cache[fname] = configs


def create_filename_key_gemm(nargs, keys):
    """
    example, N=7168,K=1536,device_name=K100_AI,dtype=..._float16.json:
    {
        '10': ...
    }
    key = M
    """
    _keys = [o for o in keys if o != "M"]
    device_name = get_gpu_label()
    dtype = get_dtypes(nargs)
    tune_key_str = ','.join([f"{o}={nargs[o]}" for o in _keys if o in nargs])
    fn = f"{tune_key_str},device_name={device_name},dtype={dtype}.json"
    return fn, str(nargs['M'])


def default_create_filename_key(nargs, keys):
    """
    example, device_name=K100_AI,dtype=float16_int32.json:
    {
        '(10, 512, 3)': ...
    }
    '(10, 512, 3)' is the original autotune key, dtype are hosited to filename.
    """
    device_name = get_gpu_label()
    dtype = get_dtypes(nargs)
    fn = f"device_name={device_name},dtype={dtype}.json"
    kval = [nargs[o] for o in keys]
    key = str(tuple(kval)) if len(kval) > 1 else str(kval[0])
    return fn, key


def save_config_gemm(dirname, config, keys, create_filename_and_key=None, **kwargs):
    """
    save best config to files, like
    https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe/configs.

    you can change filename and key through create_filename_and_key().

    :param dirname: output dir path
    :type dirname: str
    :param config: best config
    :type config: triton.Config
    :param keys: hcutune's key list
    :type keys: []
    :param create_filename_and_key: callable, create config filename and key from saving config to files
    :type create_filename_and_key: nargs: {}, keys: []
    """
    if create_filename_and_key is None:
        create_filename_and_key = create_filename_key_gemm
    return _save_config(dirname, config, keys, create_filename_and_key, **kwargs)


def default_save_config(dirname, config, keys, create_filename_and_key=None, **kwargs):
    """
    save best config to config.json in output dir, like

    N=14336,device_name=K100_AI,dtype=float16_int32.json:
    {
        '(10, 512)': {
            BLOCK_SIZE_M: 16, num_warps: 2, num_ctas: 1, num_stages: 1, maxnreg: None
        }
        ...
    }
    '(10, 512)' are the autotune keys.

    you can change filename and key through create_filename_and_key(), hosits "E" key
    to filename, like
    E=10,N=14336,device_name=K100_AI,dtype=float16_int32.json:
    {
        '512': {
            BLOCK_SIZE_M: 16, num_warps: 2, num_ctas: 1, num_stages: 1, maxnreg: None
        }
        ...
    }

    :param dirname: output dir path
    :type dirname: str
    :param config: best config
    :type config: triton.Config
    :param keys: hcutune's key list
    :type keys: []
    :param create_filename_and_key: callable, create config filename and key from saving config to files
    :type create_filename_and_key: nargs: {}, keys: []
    """
    if create_filename_and_key is None:
        create_filename_and_key = default_create_filename_key
    return _save_config(dirname, config, keys, create_filename_and_key, **kwargs)

r"""
`triton.backends.hcu.hcutune_cli` is a module that spawns up multiple distributed
processes for auto-tuning.

The utility can be used for single-process auto-tuning, in which one or
more processes will be spawned to benchmark cases.

This utility will launch the given number of processes (``--nproc``), each process
is responsible for handling a portion of the cases, including operations such as
auto-tuning and benchmarking.

**How to use this module:**

::

    >>> hcutune_cli --nproc=8 YOUR_PYTHON_SCRIPT.py [args]


**Important Notices:**

This utility must be work with triton.utils.dist_perf_report,
which should be used to replace @triton.testing.perf_report decorator in YOUR_PYTHON_SCRIPT.py.
"""

import sys
import subprocess
import os
import time
import torch
from argparse import ArgumentParser, REMAINDER


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Triton distributed hcu auto-tuning "
                                        "helper utility that will spawn up "
                                        "multiple processes")

    # optional
    parser.add_argument("--nproc", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "each process is responsible for handling a portion of the cases.")
    parser.add_argument("--devices", type=str,
                        help="Set GPU devices for benchmarking, each process is responsible for handling a portion of the cases.")
    parser.add_argument("--compile-only", action="store_true",
                        help="Just compile kernel, no tuning")
    parser.add_argument("--configs-sharding", action="store_true",
                        help="Enable sharding the pruned configs, each shard will be performed on a sub-process. "
                             "It is in conflict with triton.utils.dist_perf_report, and enable it on means that "
                             "triton.utils.dist_perf_report(x_vals/cases sharding) is disabled and replaced by "
                             "triton.testing.perf_report")
    parser.add_argument("--always-tuning", action="store_true",
                        help="Enable TRITON_HCUTUNE_ALWAYS_TUNING env variable to disable best config cache. "
                             "Hcutuner restores the best config from cache before tuning to avoid re-tuning "
                             "the existing configs. Disable this operation when this option is enabled.")
    parser.add_argument("--force", action="store_true", help="Do it in spite of warnings.")

    # positional
    parser.add_argument("src", type=str,
                        help="The full path to the single-process "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the script")

    # rest from the program
    parser.add_argument('src_args', nargs=REMAINDER)
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total = int(seconds)
    frac_ms = int(round((seconds - total) * 1000))

    m, s = divmod(total, 60)
    h, m = divmod(m, 60)

    if frac_ms == 1000:
        frac_ms = 0
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1

    if h > 0:
        return f"{h} hour{'s' if h != 1 else ''}, {m} minute{'s' if m != 1 else ''}, {s}.{frac_ms:03d} second{'s' if s != 1 else ''}"
    elif m > 0:
        return f"{m} minute{'s' if m != 1 else ''}, {s}.{frac_ms:03d} second{'s' if s != 1 else ''}"
    else:
        return f"{s}.{frac_ms:03d} second{'s' if s != 1 else ''}"


def main():
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc

    # set environmental variables for processes
    current_env = os.environ.copy()
    autotuner_env = {}
    autotuner_env["TRITON_HCUTUNE"] = "1"
    autotuner_env["TRITON_HCUTUNE_WORLD_SIZE"] = str(dist_world_size)

    devices = args.devices.split(",") if args.devices else \
                [str(i) for i in range(min(args.nproc, torch.cuda.device_count()))]

    if args.compile_only:
        autotuner_env["TRITON_HCUTUNE_COMPILE_ONLY"] = "1"
        autotuner_env["TRITON_HCUTUNE_CONFIGS_SHARDING"] = "1"
    else:
        assert args.nproc <= len(devices) or args.force, \
                "ERROR: The number of tuning processes exceeds the number of available devices " \
                f"({args.nproc} vs. {len(devices)}), which may cause tuning results to be affected " \
                "by multiple processes preempting the same device resource. " \
                "You can continue with the --force option."

    if args.configs_sharding:
        autotuner_env["TRITON_HCUTUNE_CONFIGS_SHARDING"] = "1"

    if args.always_tuning:
        autotuner_env["TRITON_HCUTUNE_ALWAYS_TUNING"] = "1"

    autotuner_env["TRITON_PRINT_AUTOTUNING"] = "1"

    current_env.update(autotuner_env)

    start = time.perf_counter()
    processes = []
    for local_rank in range(0, args.nproc):
        current_env["TRITON_HCUTUNE_LOCAL_RANK"] = str(local_rank)
        current_env["CUDA_VISIBLE_DEVICES"] = devices[local_rank % len(devices)]

        # spawn the processes
        cmd = [sys.executable, "-u"]

        cmd.append(args.src)
        cmd.extend(args.src_args)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)

    duration = time.perf_counter() - start

    print('\n' + '*' * 72)
    print(f'*  {" ".join([f"{k}={v}" for k, v in autotuner_env.items()])} {" ".join([o for o in cmd])}')
    print(f"*  Done.  Elaspsed Time: {format_duration(duration)}")
    print('*')
    print('*' * 72 + '\n')


if __name__ == "__main__":
    main()

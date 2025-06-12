r"""
`triton.utils.distributed.launch` is a module that spawns up multiple distributed
processes for auto-tuning.

The utility can be used for single-process auto-tuning, in which one or
more processes will be spawned to benchmark cases.

This utility will launch the given number of processes (``--nproc``), each process
is responsible for handling a portion of the cases, including operations such as
auto-tuning and benchmarking.

**How to use this module:**

::

    >>> python -m triton.tools.launch --nproc=8
            YOUR_PYTHON_SCRIPT.py (all arguments of your python script)


**Important Notices:**

This utility must be work with triton.utils.dist_perf_report,
which should be used to replace @triton.testing.perf_report decorator in YOUR_PYTHON_SCRIPT.py.
"""

import sys
import subprocess
import os
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
    parser.add_argument("--devices", type=str, default="0",
                        help="Set GPU devices for benchmarking, each process is responsible for handling a portion of the cases.")

    # positional
    parser.add_argument("src", type=str,
                        help="The full path to the single-process "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the script")

    # rest from the program
    parser.add_argument('src_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc

    # set environmental variables for processes
    current_env = os.environ.copy()
    current_env["TRITON_HCUTUNE_WORLD_SIZE"] = str(dist_world_size)

    processes = []
    for local_rank in range(0, args.nproc):
        current_env["TRITON_HCUTUNE_LOCAL_RANK"] = str(local_rank)

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


if __name__ == "__main__":
    main()

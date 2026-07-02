#!/usr/bin/env python3
"""Run the HCU WASP pass pipeline for lit FileCheck tests."""

import argparse
import sys

from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.compiler import make_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mlir", help="pre-WASP ttgir input file")
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--wdra-enabled", action="store_true")
    parser.add_argument("--wasp-num-load-warps", type=int, default=4)
    parser.add_argument("--wasp-num-mma-warps", type=int, default=4)
    args = parser.parse_args()

    target = GPUTarget("hip", "gfx946", 64)
    backend = make_backend(target)
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)

    module = ir.parse_mlir_module(args.input_mlir, ctx)
    pm = ir.pass_manager(ctx)
    passes.ttgpuir.add_warp_specialize_hcu(
        pm,
        args.num_stages,
        args.wdra_enabled,
        args.wasp_num_load_warps,
        args.wasp_num_mma_warps,
    )
    pm.run(module, "hcu_gemm_wasp_wdra_lit")
    sys.stdout.write(module.str_nodebug())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

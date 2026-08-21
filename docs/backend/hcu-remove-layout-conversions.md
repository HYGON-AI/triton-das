# HCU RemoveLayoutConversions enhancements

## Overview

This change ports FlagTree's enhanced `RemoveLayoutConversions` (RLC) solver
to the Triton 3.6 HCU backend as a separate
`RemoveLayoutConversionsEnhanced` pass. The existing
`RemoveLayoutConversions.cpp` and its pass remain unchanged. The HCU pipeline
selects the new pass only when the runtime switch is enabled; otherwise it
continues to instantiate the original pass. The enhanced implementation
improves layout selection and rematerialization in four stages:

1. Estimate conversion costs when resolving competing blocked layouts.
2. Propagate layout requirements backwards through rematerializable values.
3. Jointly solve small connected components instead of resolving each value
   independently.
4. Rematerialize store address and mask expressions in the stored value's
   layout, eliminating conversions at final writeback boundaries.

The fourth stage is the main source of improvement for the layout-heavy
fusion kernels evaluated during this port.

## Compatibility and rollout

The enhanced pass is always compiled into Triton. The only selection switch is
the cache-invalidating `TRITON_HCU_RLC_ENHANCE` environment variable. Enable
the enhanced pass for HCU compilation with:

```bash
export TRITON_HCU_RLC_ENHANCE=1
```

Leaving the variable unset, or setting it to `0`, preserves the existing RLC
pipeline. The environment variable is cache-invalidating, so enabled and
disabled compilations cannot accidentally reuse the same cached TTGIR or
HSACO.

All HCU RLC invocation points use the same runtime switch, including the
passes before and after matmul acceleration, MLS encoding insertion,
instruction scheduling, in-thread transpose, and warp specialization.

## HCU-specific behavior retained by the port

The FlagTree implementation was merged with, rather than substituted for,
the existing HCU RLC implementation. In particular, the port retains:

- `MatrixLoadOp` as a layout anchor and a non-rematerializable operation;
- preference for `DotOperandEncoding` on matrix-load conflict resolution;
- the gfx938 matrix-load mask pattern used by the
  `matrix_load -> select` workaround; and
- the existing TLE hard-encoding behavior when TLE is enabled.

## Validation

The port was built against the LLVM revision pinned by the Triton 3.6 branch,
`ddb5c4b4bddd641e1afabf70ef1b777fca9ea150`. The complete Triton compiler,
Python extension, `triton-opt`, and HCU backend built successfully.

GPU validation used an HCU BW151 and compared disabled and enabled builds on
the same device. Both tested fusion kernels passed their PyTorch reference
checks.

| Kernel | `convert_layout` disabled | enabled | Time disabled | enabled | HSACO disabled | enabled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fused layer norm + sigmoid | 13 | 0 | 0.03070 ms | 0.01497 ms | 60,816 B | 36,064 B |
| fused layer norm | 15 | 0 | 0.00783 ms | 0.00755 ms | 14,576 B | 13,000 B |

For a regular GEMM retaining one required conversion, the generated HSACO was
byte-for-byte identical with the enhancement disabled and enabled. Likewise,
freshly generated four-warp versions of the fusion kernels already contained
zero conversions in the baseline and produced identical HSACO. The benefit
therefore applies to kernels that reach RLC with redundant final writeback
conversions; the switch does not imply a universal speedup.

## Testing an individual kernel

Use distinct cache directories when comparing the generated artifacts. This
also makes conversion counts easy to inspect:

```bash
TRITON_HCU_RLC_ENHANCE=0 TRITON_CACHE_DIR=/tmp/rlc-off python kernel.py
TRITON_HCU_RLC_ENHANCE=1 TRITON_CACHE_DIR=/tmp/rlc-on  python kernel.py

rg -o 'ttg.convert_layout' /tmp/rlc-off -g '*.ttgir' | wc -l
rg -o 'ttg.convert_layout' /tmp/rlc-on  -g '*.ttgir' | wc -l
```

Correctness tests should be run before performance comparisons because the
store-layout phase can duplicate address or mask computations while removing
a conversion boundary.

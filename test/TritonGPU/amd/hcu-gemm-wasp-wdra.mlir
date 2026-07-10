// Verify GEMM WASP / WASP+WDRA lowering IR captured from:
//   triton-opt %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir \
//     --tritongpu-automatic-warp-specialization-hcu="num-stages=2 wdra-enabled=false wasp-num-load-warps=4 wasp-num-mma-warps=4"
//   triton-opt %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir \
//     --tritongpu-automatic-warp-specialization-hcu="num-stages=2 wdra-enabled=false wasp-num-load-warps=4 wasp-num-mma-warps=8"
//   triton-opt %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir \
//     --tritongpu-automatic-warp-specialization-hcu="num-stages=2 wdra-enabled=true wasp-num-load-warps=4 wasp-num-mma-warps=4"
//   triton-opt %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir \
//     --tritongpu-automatic-warp-specialization-hcu="num-stages=2 wdra-enabled=true wasp-num-load-warps=4 wasp-num-mma-warps=8"
//
// Note: triton-opt currently cannot emit ROCDL HCU abarrier ops without loading
// the HCU dialect via the Python backend; goldens are produced with the HCU
// Python pass manager and checked here with FileCheck.

// RUN: FileCheck %s --check-prefix=WASP44 --input-file=%S/Inputs/hcu-gemm-wasp-only-post-44.ttgir
// RUN: FileCheck %s --check-prefix=WASP48 --input-file=%S/Inputs/hcu-gemm-wasp-only-post-48.ttgir
// RUN: FileCheck %s --check-prefix=WDRA44 --input-file=%S/Inputs/hcu-gemm-wasp-wdra-post-44.ttgir
// RUN: FileCheck %s --check-prefix=WDRA48 --input-file=%S/Inputs/hcu-gemm-wasp-wdra-post-48.ttgir

// WASP-only 4 load + 4 MMA: 2 partitions, no MMA split.
// WASP44: module attributes
// WASP44-SAME: ttg.total-num-warps" = 8 : i32
// WASP44-LABEL: @matmul_kernel
// WASP44: ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16
// WASP44: rocdl.hcu.s.abarrier.init
// WASP44: ttg.warp_specialize
// WASP44-SAME: requestedRegisters = array<i32: 88, 88>
// WASP44-SAME: warpGroupStartIds = array<i32: 0, 4>
// WASP44: partition0
// WASP44-SAME: num_warps(4)
// WASP44: rocdl.hcu.s.abarrier.try.wait
// WASP44: rocdl.hcu.s.abarrier.arrive
// WASP44: partition1
// WASP44-SAME: num_warps(4)
// WASP44-NOT: partition2

// WASP-only 4 load + 8 MMA: still 2 partitions (MMA stays one group of 8).
// WASP48: module attributes
// WASP48-SAME: ttg.total-num-warps" = 12 : i32
// WASP48-LABEL: @matmul_kernel
// WASP48: ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16
// WASP48: rocdl.hcu.s.abarrier.init
// WASP48: ttg.warp_specialize
// WASP48-SAME: requestedRegisters = array<i32: 88, 88>
// WASP48-SAME: warpGroupStartIds = array<i32: 0, 8>
// WASP48: partition0
// WASP48-SAME: num_warps(8)
// WASP48: partition1
// WASP48-SAME: num_warps(4)
// WASP48-NOT: partition2

// WASP+WDRA 4+4: 2 partitions with WDRA register requests.
// WDRA44: module attributes
// WDRA44-SAME: ttg.total-num-warps" = 8 : i32
// WDRA44-LABEL: @matmul_kernel
// WDRA44: ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16
// WDRA44: rocdl.hcu.s.abarrier.init
// WDRA44: ttg.warp_specialize
// WDRA44-SAME: requestedRegisters = array<i32: 88, 88>
// WDRA44-SAME: warpGroupStartIds = array<i32: 0, 4>
// WDRA44: partition0
// WDRA44-SAME: num_warps(4)
// WDRA44: rocdl.hcu.s.abarrier.try.wait
// WDRA44: rocdl.hcu.s.abarrier.arrive
// WDRA44: partition1
// WDRA44-SAME: num_warps(4)
// WDRA44-NOT: partition2

// WASP+WDRA 4+8: SplitMma creates 3 partitions.
// WDRA48: module attributes
// WDRA48-SAME: ttg.total-num-warps" = 12 : i32
// WDRA48-LABEL: @matmul_kernel
// WDRA48: ttg.warp_specialize
// WDRA48-SAME: requestedRegisters = array<i32: 88, 88, 88>
// WDRA48-SAME: warpGroupStartIds = array<i32: 0, 4, 8>
// WDRA48: partition0
// WDRA48-SAME: num_warps(4)
// WDRA48: partition1
// WDRA48: partition2

// Verify WASP+WDRA GEMM lowering IR captured from:
//   triton-opt %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir \
//     --tritongpu-automatic-warp-specialization-hcu="num-stages=2 wdra-enabled=true wasp-num-load-warps=4 wasp-num-mma-warps=4"
//   triton-opt %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir \
//     --tritongpu-automatic-warp-specialization-hcu="num-stages=2 wdra-enabled=true wasp-num-load-warps=4 wasp-num-mma-warps=8"

// RUN: FileCheck %s --check-prefix=WASP44 --input-file=%S/Inputs/hcu-gemm-wasp-wdra-post-44.ttgir
// RUN: FileCheck %s --check-prefix=WASP48 --input-file=%S/Inputs/hcu-gemm-wasp-wdra-post-48.ttgir

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

// WASP48: module attributes
// WASP48-SAME: ttg.total-num-warps" = 12 : i32
// WASP48-LABEL: @matmul_kernel
// WASP48: ttg.warp_specialize
// WASP48-SAME: requestedRegisters = array<i32: 88, 88, 88>
// WASP48-SAME: warpGroupStartIds = array<i32: 0, 4, 8>
// WASP48: partition0
// WASP48-SAME: num_warps(4)
// WASP48: partition1
// WASP48: partition2

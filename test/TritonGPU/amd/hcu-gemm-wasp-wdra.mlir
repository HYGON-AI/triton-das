// RUN: %PYTHON %S/run_hcu_wasp_pass.py %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir --wdra-enabled --wasp-num-load-warps=4 --wasp-num-mma-warps=4 | FileCheck %s --check-prefix=WASP44
// RUN: %PYTHON %S/run_hcu_wasp_pass.py %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir --wdra-enabled --wasp-num-load-warps=4 --wasp-num-mma-warps=8 | FileCheck %s --check-prefix=WASP48

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

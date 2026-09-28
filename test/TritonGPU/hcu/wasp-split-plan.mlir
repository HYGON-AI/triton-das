// RUN: triton-opt %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir --tritonhcu-prepare-wasp-split-plan | FileCheck %s --check-prefix=PLAN
// RUN: sed 's/module attributes {/module attributes {hcu.wasp_partition_warps = array<i32: 4, 4>,/' %S/Inputs/hcu-gemm-wasp-wdra-pre.mlir | triton-opt --tritongpu-data-partition | FileCheck %s --check-prefix=UNSPLIT

// The early plan is a WASP contract, independent of WDRA register allocation.
// PLAN: tt.dot {{.*}}hcu.wasp_split = array<i32: 0, 2, 0>
// PLAN: hcu.wasp_split_policy = 2 : i32

// Explicit OnePOneC must be a no-op even when DataPartition is run directly:
// the unscheduled full-tile IR must not enter the slicing machinery.
// UNSPLIT: module attributes {hcu.wasp_partition_warps = array<i32: 4, 4>
// UNSPLIT: tt.dot {{.*}} -> tensor<128x128xf32,
// UNSPLIT-NOT: tt.dot
// UNSPLIT: tt.return

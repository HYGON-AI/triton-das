// FileCheck pre-generated AMDGCN for clang scheduling knobs on
// Inputs/hcu-misched-max-ilp-sched.ll:
//   max_ilp        : -mllvm=-amdgpu-enable-max-ilp-scheduling-strategy=true
//   no_regpressure : -mllvm=-misched-regpressure=false
//
// Goldens are offline; regenerate from this directory (test/TritonGPU/hcu/):
//
//   CLANG=/opt/dtk/aillvm/bin/clang-18
//   IN=Inputs/hcu-misched-max-ilp-sched.ll
//   OUT=Inputs/hcu-misched-max-ilp-sched
//   BASE=(-target amdgcn-amd-amdhsa -mcpu=gfx936:xnack-
//         -mllvm=-check-valu-data-forward-hazards=0
//         -mllvm=-disable-cluster-lds-memops=true
//         -mllvm=-hcu-pre-emit-load-store-opt=false
//         -mllvm=-support-768-vgprs=true
//         -mllvm=-enable-hcu-approx-func-fp-math=true
//         -mllvm=-hcu-update-wait-by-reverse-search=true
//         -O3)
//   MAXILP=-mllvm=-amdgpu-enable-max-ilp-scheduling-strategy=true
//   NOREG=-mllvm=-misched-regpressure=false
//   $CLANG "${BASE[@]}"                "$IN" -S -o "$OUT-baseline.amdgcn"
//   $CLANG "${BASE[@]}" $MAXILP        "$IN" -S -o "$OUT-max-ilp.amdgcn"
//   $CLANG "${BASE[@]}" $NOREG         "$IN" -S -o "$OUT-no-regpressure.amdgcn"
//   $CLANG "${BASE[@]}" $MAXILP $NOREG "$IN" -S -o "$OUT-max-ilp-no-regpressure.amdgcn"
//
// RUN: FileCheck %s --check-prefix=BASE   --input-file=%S/Inputs/hcu-misched-max-ilp-sched-baseline.amdgcn
// RUN: FileCheck %s --check-prefix=NOREG  --input-file=%S/Inputs/hcu-misched-max-ilp-sched-no-regpressure.amdgcn
// RUN: FileCheck %s --check-prefix=MAXILP --input-file=%S/Inputs/hcu-misched-max-ilp-sched-max-ilp.amdgcn
// RUN: FileCheck %s --check-prefix=BOTH   --input-file=%S/Inputs/hcu-misched-max-ilp-sched-max-ilp-no-regpressure.amdgcn

// Baseline: early vmcnt wait between payload loads.
// BASE-LABEL: triton_red_fused__to_copy_cat_native_layer_norm_67:
// BASE: buffer_load_dwordx4
// BASE: s_waitcnt{{.*}}vmcnt
// BASE: buffer_load_dwordx4

// no_regpressure: still has a vmcnt wait among the leading loads.
// NOREG-LABEL: triton_red_fused__to_copy_cat_native_layer_norm_67:
// NOREG: buffer_load_dwordx4
// NOREG: s_waitcnt{{.*}}vmcnt
// NOREG: buffer_load_dwordx4

// max_ilp: long load burst with no intervening vmcnt wait.
// MAXILP-LABEL: triton_red_fused__to_copy_cat_native_layer_norm_67:
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4
// MAXILP-NOT: s_waitcnt{{.*}}vmcnt
// MAXILP: buffer_load_dwordx4

// both flags: same long uninterrupted load burst as MAXILP.
// BOTH-LABEL: triton_red_fused__to_copy_cat_native_layer_norm_67:
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4
// BOTH-NOT: s_waitcnt{{.*}}vmcnt
// BOTH: buffer_load_dwordx4

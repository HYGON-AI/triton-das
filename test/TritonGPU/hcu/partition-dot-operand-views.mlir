// RUN: triton-opt %s -tritongpu-partition-scheduling-hcu | FileCheck %s
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// The transposed K operand must be consumed only by the MMA partition.
// Putting its layout conversion in root duplicates empty-barrier arrivals.
// CHECK-LABEL: tt.func public @_flash_attention_fwd_kernel
// CHECK: tt.load {{.*}}ttg.partition = array<i32: 2>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: 1>{{.*}}tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #linear>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: 1>
// CHECK: tt.dot {{.*}}ttg.partition = array<i32: 1>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32], [16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 2, order = [0, 1]}>
#shared2 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 8], [4, 0], [8, 0], [16, 0], [0, 32]]}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {hcu.empty_arrive_after_mmac = false, hcu.sched_barrier_before_mmac = true, hcu.sched_barrier_between_a_b_loads = true, hcu.wdra_topo = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_flash_attention_fwd_kernel(%Q: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %K: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %V: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Out: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Lse: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %sm_scale: f32, %stride_qb: i32 {tt.divisibility = 16 : i32}, %stride_qh: i32 {tt.divisibility = 16 : i32}, %stride_qs: i32 {tt.divisibility = 16 : i32}, %stride_kb: i32 {tt.divisibility = 16 : i32}, %stride_kh: i32 {tt.divisibility = 16 : i32}, %stride_ks: i32 {tt.divisibility = 16 : i32}, %stride_vb: i32 {tt.divisibility = 16 : i32}, %stride_vh: i32 {tt.divisibility = 16 : i32}, %stride_vs: i32 {tt.divisibility = 16 : i32}, %stride_ob: i32 {tt.divisibility = 16 : i32}, %stride_oh: i32 {tt.divisibility = 16 : i32}, %stride_os: i32 {tt.divisibility = 16 : i32}, %stride_lb: i32 {tt.divisibility = 16 : i32}, %stride_lh: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<160> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %cst_1 = arith.constant dense<160> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %l_i = arith.constant dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_2 = arith.constant dense<0.693147182> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_3 = arith.constant dense<0xFF800000> : tensor<64x32xf32, #mma>
    %m_i = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c32_i32 = arith.constant 32 : i32
    %c160_i32 = arith.constant 160 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_5 = arith.constant 1.44269502 : f32
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #blocked>
    %q = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %c3_i32 = arith.constant 3 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_7 = arith.constant dense<160> : tensor<64xi32, #blocked1>
    %cst_8 = arith.constant dense<160> : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %q_valid = arith.constant dense<160> : tensor<64xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %pid = tt.get_program_id x : i32
    %query_block = arith.remsi %pid, %c3_i32 : i32
    %head_batch = arith.divsi %pid, %c3_i32 : i32
    %offs_m = arith.muli %query_block, %c64_i32 : i32
    %offs_m_9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_m_10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_m_12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %offs_m_14 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_m_15 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_16 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_m_17 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_18 = tt.splat %offs_m : i32 -> tensor<64xi32, #blocked1>
    %offs_m_19 = arith.addi %offs_m_14, %offs_m_9 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_m_20 = arith.addi %offs_m_15, %offs_m_10 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_21 = arith.addi %offs_m_16, %offs_m_11 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %offs_m_22 = arith.addi %offs_m_17, %offs_m_12 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_23 = arith.addi %offs_m_18, %offs_m_13 : tensor<64xi32, #blocked1>
    %offs_n = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_24 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_n_25 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %q_valid_26 = arith.cmpi slt, %offs_m_21, %q_valid : tensor<64xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %q_valid_27 = arith.cmpi slt, %offs_m_22, %cst_8 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %q_valid_28 = arith.cmpi slt, %offs_m_23, %cst_7 : tensor<64xi32, #blocked1>
    %q_ptrs = arith.muli %head_batch, %stride_qb : i32
    %q_ptrs_29 = tt.addptr %Q, %q_ptrs : !tt.ptr<f16>, i32
    %q_ptrs_30 = tt.expand_dims %offs_m_19 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xi32, #mma>
    %q_ptrs_31 = tt.expand_dims %offs_m_20 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %q_ptrs_32 = tt.splat %stride_qs : i32 -> tensor<64x1xi32, #blocked>
    %q_ptrs_33 = arith.muli %q_ptrs_31, %q_ptrs_32 : tensor<64x1xi32, #blocked>
    %q_ptrs_34 = tt.splat %q_ptrs_29 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %q_ptrs_35 = tt.addptr %q_ptrs_34, %q_ptrs_33 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %q_ptrs_36 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %q_ptrs_37 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %q_ptrs_38 = tt.expand_dims %q_ptrs_36 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %q_ptrs_39 = tt.expand_dims %q_ptrs_37 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %q_ptrs_40 = tt.broadcast %q_ptrs_35 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %q_ptrs_41 = tt.broadcast %q_ptrs_38 : tensor<1x64xi32, #mma> -> tensor<64x64xi32, #mma>
    %q_ptrs_42 = tt.broadcast %q_ptrs_39 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %q_ptrs_43 = tt.addptr %q_ptrs_40, %q_ptrs_42 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %q_44 = tt.expand_dims %q_valid_26 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xi1, #mma>
    %q_45 = tt.expand_dims %q_valid_27 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
    %q_46 = tt.broadcast %q_44 : tensor<64x1xi1, #mma> -> tensor<64x64xi1, #mma>
    %q_47 = tt.broadcast %q_45 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
    %q_48 = tt.load %q_ptrs_43, %q_47, %q : tensor<64x64x!tt.ptr<f16>, #blocked>
    %q_49 = arith.mulf %sm_scale, %cst_5 : f32
    %q_50 = arith.extf %q_48 : tensor<64x64xf16, #blocked> to tensor<64x64xf32, #blocked>
    %q_51 = tt.splat %q_49 : f32 -> tensor<64x64xf32, #blocked>
    %q_52 = arith.mulf %q_50, %q_51 : tensor<64x64xf32, #blocked>
    %q_53 = arith.truncf %q_52 : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %q_54 = ttg.local_alloc %q_53 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %q_55 = ttg.local_load %q_54 : !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %k_ptrs = arith.muli %head_batch, %stride_kb : i32
    %k_ptrs_56 = tt.addptr %K, %k_ptrs : !tt.ptr<f16>, i32
    %k_ptrs_57 = tt.splat %stride_ks : i32 -> tensor<32x1xi32, #blocked>
    %k_ptrs_58 = tt.splat %k_ptrs_56 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %k_ptrs_59 = tt.broadcast %q_ptrs_39 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %v_ptrs = arith.muli %head_batch, %stride_vb : i32
    %v_ptrs_60 = tt.addptr %V, %v_ptrs : !tt.ptr<f16>, i32
    %v_ptrs_61 = tt.splat %stride_vs : i32 -> tensor<32x1xi32, #blocked>
    %v_ptrs_62 = tt.splat %v_ptrs_60 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %acc:3 = scf.for %acc_75 = %c0_i32 to %c160_i32 step %c32_i32 iter_args(%m_i_76 = %m_i, %l_i_77 = %l_i, %arg23 = %cst_4) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64xf32, #mma>)  : i32 {
      %token = tt.splat %acc_75 {ttg.partition = array<i32: 0>} : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %token_78 = tt.splat %acc_75 {ttg.partition = array<i32: 0>} : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %token_79 = tt.splat %acc_75 {ttg.partition = array<i32: 0>} : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %token_80 = arith.addi %token, %offs_n {ttg.partition = array<i32: 0>} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %token_81 = arith.addi %token_78, %offs_n_24 {ttg.partition = array<i32: 0>} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %token_82 = arith.addi %token_79, %offs_n_25 {ttg.partition = array<i32: 0>} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %kv_valid = arith.cmpi slt, %token_81, %cst_1 {ttg.partition = array<i32: 0>} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %kv_valid_83 = arith.cmpi slt, %token_82, %cst {ttg.partition = array<i32: 0>} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %k_ptrs_84 = tt.expand_dims %token_80 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
      %k_ptrs_85 = arith.muli %k_ptrs_84, %k_ptrs_57 {ttg.partition = array<i32: 0>} : tensor<32x1xi32, #blocked>
      %k_ptrs_86 = tt.addptr %k_ptrs_58, %k_ptrs_85 {ttg.partition = array<i32: 0>} : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
      %k_ptrs_87 = tt.broadcast %k_ptrs_86 {ttg.partition = array<i32: 0>} : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x64x!tt.ptr<f16>, #blocked>
      %k_ptrs_88 = tt.addptr %k_ptrs_87, %k_ptrs_59 {ttg.partition = array<i32: 0>} : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
      %v_ptrs_89 = arith.muli %k_ptrs_84, %v_ptrs_61 {ttg.partition = array<i32: 0>} : tensor<32x1xi32, #blocked>
      %v_ptrs_90 = tt.addptr %v_ptrs_62, %v_ptrs_89 {ttg.partition = array<i32: 0>} : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
      %v_ptrs_91 = tt.broadcast %v_ptrs_90 {ttg.partition = array<i32: 0>} : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x64x!tt.ptr<f16>, #blocked>
      %v_ptrs_92 = tt.addptr %v_ptrs_91, %k_ptrs_59 {ttg.partition = array<i32: 0>} : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
      %k = tt.expand_dims %kv_valid {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
      %k_93 = tt.broadcast %k {ttg.partition = array<i32: 0>} : tensor<32x1xi1, #blocked> -> tensor<32x64xi1, #blocked>
      %k_94 = tt.load %k_ptrs_88, %k_93, %cst_6 {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<32x64x!tt.ptr<f16>, #blocked>
      %scores = ttg.convert_layout %k_94 {ttg.partition = array<i32: 0>} : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #linear>
      %scores_95 = tt.trans %scores {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<32x64xf16, #linear> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %scores_96 = tt.dot %q_55, %scores_95, %cst_0, inputPrecision = tf32 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x32xf32, #mma>
      %valid = tt.expand_dims %kv_valid_83 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xi1, #mma>
      %scores_97 = tt.broadcast %valid {ttg.partition = array<i32: 0>} : tensor<1x32xi1, #mma> -> tensor<64x32xi1, #mma>
      %scores_98 = arith.select %scores_97, %scores_96, %cst_3 {ttg.partition = array<i32: 0>} : tensor<64x32xi1, #mma>, tensor<64x32xf32, #mma>
      %m_ij = "tt.reduce"(%scores_98) <{axis = 1 : i32}> ({
      ^bb0(%m_ij_116: f32, %m_ij_117: f32):
        %m_ij_118 = arith.maxnumf %m_ij_116, %m_ij_117 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %m_ij_118 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>} : (tensor<64x32xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %m_ij_99 = arith.maxnumf %m_i_76, %m_ij {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %alpha = arith.subf %m_i_76, %m_ij_99 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %alpha_100 = math.exp2 %alpha {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %p = tt.expand_dims %m_ij_99 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
      %p_101 = tt.broadcast %p {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #mma> -> tensor<64x32xf32, #mma>
      %p_102 = arith.subf %scores_98, %p_101 {ttg.partition = array<i32: 0>} : tensor<64x32xf32, #mma>
      %p_103 = math.exp2 %p_102 {ttg.partition = array<i32: 0>} : tensor<64x32xf32, #mma>
      %l_i_104 = arith.mulf %l_i_77, %alpha_100 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %l_i_105 = "tt.reduce"(%p_103) <{axis = 1 : i32}> ({
      ^bb0(%l_i_116: f32, %l_i_117: f32):
        %l_i_118 = arith.addf %l_i_116, %l_i_117 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %l_i_118 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>} : (tensor<64x32xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %l_i_106 = arith.addf %l_i_104, %l_i_105 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %v = tt.load %v_ptrs_92, %k_93, %cst_6 {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<32x64x!tt.ptr<f16>, #blocked>
      %acc_107 = tt.expand_dims %alpha_100 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
      %acc_108 = tt.broadcast %acc_107 {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #mma> -> tensor<64x64xf32, #mma>
      %acc_109 = arith.mulf %arg23, %acc_108 {ttg.partition = array<i32: 0>} : tensor<64x64xf32, #mma>
      %acc_110 = arith.truncf %p_103 {ttg.partition = array<i32: 0>} : tensor<64x32xf32, #mma> to tensor<64x32xf16, #mma>
      %acc_111 = ttg.local_alloc %acc_110 {ttg.partition = array<i32: 0>} : (tensor<64x32xf16, #mma>) -> !ttg.memdesc<64x32xf16, #shared1, #smem>
      %acc_112 = ttg.local_load %acc_111 {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x32xf16, #shared1, #smem> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %acc_113 = ttg.local_alloc %v {ttg.partition = array<i32: 0>} : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared2, #smem>
      %acc_114 = ttg.local_load %acc_113 {ttg.partition = array<i32: 0>} : !ttg.memdesc<32x64xf16, #shared2, #smem> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %acc_115 = tt.dot %acc_112, %acc_114, %acc_109, inputPrecision = tf32 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      scf.yield {ttg.partition = array<i32: 0>} %m_ij_99, %l_i_106, %acc_115 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64xf32, #mma>
    } {hcu.wdra_split_policy = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
    %out = tt.expand_dims %acc#1 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
    %out_63 = tt.broadcast %out : tensor<64x1xf32, #mma> -> tensor<64x64xf32, #mma>
    %out_64 = arith.divf %acc#2, %out_63 : tensor<64x64xf32, #mma>
    %out_ptrs = arith.muli %head_batch, %stride_ob : i32
    %out_ptrs_65 = tt.addptr %Out, %out_ptrs : !tt.ptr<f16>, i32
    %out_ptrs_66 = tt.splat %stride_os : i32 -> tensor<64x1xi32, #mma>
    %out_ptrs_67 = arith.muli %q_ptrs_30, %out_ptrs_66 : tensor<64x1xi32, #mma>
    %out_ptrs_68 = tt.splat %out_ptrs_65 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #mma>
    %out_ptrs_69 = tt.addptr %out_ptrs_68, %out_ptrs_67 : tensor<64x1x!tt.ptr<f16>, #mma>, tensor<64x1xi32, #mma>
    %out_ptrs_70 = tt.broadcast %out_ptrs_69 : tensor<64x1x!tt.ptr<f16>, #mma> -> tensor<64x64x!tt.ptr<f16>, #mma>
    %out_ptrs_71 = tt.addptr %out_ptrs_70, %q_ptrs_41 : tensor<64x64x!tt.ptr<f16>, #mma>, tensor<64x64xi32, #mma>
    %0 = arith.truncf %out_64 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
    tt.store %out_ptrs_71, %0, %q_46 : tensor<64x64x!tt.ptr<f16>, #mma>
    %lse_ptrs = arith.muli %head_batch, %stride_lb : i32
    %lse_ptrs_72 = tt.addptr %Lse, %lse_ptrs : !tt.ptr<f32>, i32
    %lse_ptrs_73 = tt.splat %lse_ptrs_72 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
    %lse_ptrs_74 = tt.addptr %lse_ptrs_73, %offs_m_23 : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
    %1 = math.log2 %acc#1 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %2 = arith.addf %acc#0, %1 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %3 = arith.mulf %2, %cst_2 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %4 = ttg.convert_layout %3 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64xf32, #blocked1>
    tt.store %lse_ptrs_74, %4, %q_valid_28 : tensor<64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// Modified by Hygon Information Technology Co., Ltd., 2026.
// RUN: triton-opt %s -tritonhcu-prepare-block-pingpong | FileCheck %s
// RUN: triton-opt %s -tritonhcu-prepare-block-pingpong -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline -canonicalize | FileCheck %s --check-prefix=PIPELINE
// RUN: triton-opt %s -tritonhcu-prepare-block-pingpong -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline -canonicalize -tritonhcu-block-pingpong -triton-loop-unroll -canonicalize -tritonhcu-finalize-block-pingpong | FileCheck %s --check-prefix=INTEGRATED

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked_a_bf16_bk64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_b_bf16_bk64 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked_a_fp8 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_a_fp8_bk128 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_b_fp8 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked_b_fp8_bk128 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#mma_n = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#dot_a_n = #ttg.dot_op<{opIdx = 0, parent = #mma_n, kWidth = 4}>
#dot_b_n = #ttg.dot_op<{opIdx = 1, parent = #mma_n, kWidth = 4}>
#mma_fp8 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#dot_a_fp8 = #ttg.dot_op<{opIdx = 0, parent = #mma_fp8, kWidth = 8}>
#dot_b_fp8 = #ttg.dot_op<{opIdx = 1, parent = #mma_fp8, kWidth = 8}>
#mma_fp8_n = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#dot_a_fp8_n = #ttg.dot_op<{opIdx = 0, parent = #mma_fp8_n, kWidth = 8}>
#dot_b_fp8_n = #ttg.dot_op<{opIdx = 1, parent = #mma_fp8_n, kWidth = 8}>
#mma_fp8_legacy = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 1, tilesPerWarp = [1, 2]}>
#dot_a_fp8_legacy = #ttg.dot_op<{opIdx = 0, parent = #mma_fp8_legacy, kWidth = 8}>
#dot_b_fp8_legacy = #ttg.dot_op<{opIdx = 1, parent = #mma_fp8_legacy, kWidth = 8}>
#mma_fp8_n_legacy = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 1, tilesPerWarp = [1, 2]}>
#dot_a_fp8_n_legacy = #ttg.dot_op<{opIdx = 0, parent = #mma_fp8_n_legacy, kWidth = 8}>
#dot_b_fp8_n_legacy = #ttg.dot_op<{opIdx = 1, parent = #mma_fp8_n_legacy, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // Native FP8 needs BK64 so each BlockPingpong half is one complete K32 MMAC
  // tile. Verify both preparation and the integrated four-phase transform.
  // CHECK-LABEL: tt.func public @fp8_bk64_candidate
  // INTEGRATED-LABEL: tt.func public @fp8_bk64_candidate
  // INTEGRATED: tt.dot {{.*}} tensor<256x32xf8E4M3FN
  // INTEGRATED: ttg.local_load {{.+}} -> tensor<256x32xf8E4M3FN
  // INTEGRATED: ttg.local_load {{.+}} -> tensor<32x256xf8E4M3FN
  // INTEGRATED: gpu.barrier
  // INTEGRATED: tt.dot {{.*}} tensor<256x32xf8E4M3FN
  // INTEGRATED: } {hcu.block_pingpong.applied, hcu.block_pingpong.static_buffers
  tt.func public @fp8_bk64_candidate(
      %a_ptrs: tensor<256x64x!tt.ptr<f8E4M3FN>, #blocked_a_fp8> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<64x256x!tt.ptr<f8E4M3FN>, #blocked_b_fp8> {tt.contiguity = dense<[16, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 64 : i32})
      -> tensor<256x256xf32, #mma_fp8> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma_fp8>
    %result = scf.for %iv = %c0_i32 to %K step %c64_i32
        iter_args(%acc = %zero) -> (tensor<256x256xf32, #mma_fp8>) : i32 {
      %a = tt.load %a_ptrs : tensor<256x64x!tt.ptr<f8E4M3FN>, #blocked_a_fp8>
      %b = tt.load %b_ptrs : tensor<64x256x!tt.ptr<f8E4M3FN>, #blocked_b_fp8>
      %dot_a = ttg.convert_layout %a : tensor<256x64xf8E4M3FN, #blocked_a_fp8> -> tensor<256x64xf8E4M3FN, #dot_a_fp8>
      %dot_b = ttg.convert_layout %b : tensor<64x256xf8E4M3FN, #blocked_b_fp8> -> tensor<64x256xf8E4M3FN, #dot_b_fp8>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<256x64xf8E4M3FN, #dot_a_fp8> * tensor<64x256xf8E4M3FN, #dot_b_fp8>
          -> tensor<256x256xf32, #mma_fp8>
      scf.yield %next : tensor<256x256xf32, #mma_fp8>
    }
    // CHECK: } {hcu.block_pingpong.buffer_load_stage3
    // CHECK-SAME: tt.num_stages = 3 : i32
    // CHECK-SAME: ttg.pipeline.hcu_local_store_stage = 1 : i32
    tt.return %result : tensor<256x256xf32, #mma_fp8>
  }

  // Keep the minimum 128x128 output tile in the same native Bit8 BK64
  // schedule family.
  // CHECK-LABEL: tt.func public @fp8_128x128_candidate
  // INTEGRATED-LABEL: tt.func public @fp8_128x128_candidate
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  tt.func public @fp8_128x128_candidate(
      %a_ptrs: tensor<128x64x!tt.ptr<f8E4M3FN>, #blocked_a_fp8> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8> {tt.contiguity = dense<[16, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 64 : i32})
      -> tensor<128x128xf32, #mma_fp8> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_fp8>
    %result = scf.for %iv = %c0_i32 to %K step %c64_i32
        iter_args(%acc = %zero) -> (tensor<128x128xf32, #mma_fp8>) : i32 {
      %a = tt.load %a_ptrs : tensor<128x64x!tt.ptr<f8E4M3FN>, #blocked_a_fp8>
      %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8>
      %dot_a = ttg.convert_layout %a : tensor<128x64xf8E4M3FN, #blocked_a_fp8> -> tensor<128x64xf8E4M3FN, #dot_a_fp8>
      %dot_b = ttg.convert_layout %b : tensor<64x128xf8E4M3FN, #blocked_b_fp8> -> tensor<64x128xf8E4M3FN, #dot_b_fp8>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<128x64xf8E4M3FN, #dot_a_fp8> * tensor<64x128xf8E4M3FN, #dot_b_fp8>
          -> tensor<128x128xf32, #mma_fp8>
      scf.yield %next : tensor<128x128xf32, #mma_fp8>
    }
    // CHECK: } {hcu.block_pingpong.buffer_load_stage3
    // CHECK-SAME: tt.num_stages = 3 : i32
    tt.return %result : tensor<128x128xf32, #mma_fp8>
  }

  // A BK128 128x128 tile fits as a conventional two-A/two-B ring: four
  // independent 16 KiB physical slots use exactly 64 KiB.  It must not inherit
  // the single-A overwrite barriers required by the larger 256x128 tile.
  // CHECK-LABEL: tt.func public @fp8_bk128_128x128_double_candidate
  // CHECK: } {hcu.block_pingpong.buffer_load_stage3
  // INTEGRATED-LABEL: tt.func public @fp8_bk128_128x128_double_candidate
  // INTEGRATED-COUNT-4: ttg.local_alloc : () -> !ttg.memdesc<128x128xf8E4M3FN
  // INTEGRATED-NOT: ttg.local_barrier
  // The tuned BK128 body stores next A after dot.lo, then reads current A.hi;
  // next B and the full publication barrier stay before dot.hi.
  // INTEGRATED: gpu.barrier
  // INTEGRATED: scf.for
  // INTEGRATED: tt.dot {{.*}} tensor<128x64xf8E4M3FN
  // INTEGRATED: ttg.local_store {{.*}} -> !ttg.memdesc<128x128xf8E4M3FN, #shared
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<128x64xf8E4M3FN
  // INTEGRATED: ttg.local_store {{.*}} -> !ttg.memdesc<128x128xf8E4M3FN, #shared
  // INTEGRATED: gpu.barrier
  // INTEGRATED: tt.dot {{.*}} tensor<128x64xf8E4M3FN
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  tt.func public @fp8_bk128_128x128_double_candidate(
      %a_ptrs: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8_bk128> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8> {tt.contiguity = dense<[16, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 128 : i32})
      -> tensor<128x128xf32, #mma_fp8> {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_fp8>
    %result = scf.for %iv = %c0_i32 to %K step %c128_i32
        iter_args(%acc = %zero) -> (tensor<128x128xf32, #mma_fp8>) : i32 {
      %a = tt.load %a_ptrs : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8_bk128>
      %b = tt.load %b_ptrs : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8>
      %dot_a = ttg.convert_layout %a : tensor<128x128xf8E4M3FN, #blocked_a_fp8_bk128> -> tensor<128x128xf8E4M3FN, #dot_a_fp8>
      %dot_b = ttg.convert_layout %b : tensor<128x128xf8E4M3FN, #blocked_b_fp8> -> tensor<128x128xf8E4M3FN, #dot_b_fp8>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<128x128xf8E4M3FN, #dot_a_fp8> * tensor<128x128xf8E4M3FN, #dot_b_fp8>
          -> tensor<128x128xf32, #mma_fp8>
      scf.yield %next : tensor<128x128xf32, #mma_fp8>
    }
    tt.return %result : tensor<128x128xf32, #mma_fp8>
  }

  // A BK128 256x128 tile fits by physically reusing the 32 KiB A tile while
  // retaining two independently selected 16 KiB B slots. Use the gfx936
  // LEGACY MMAC spelling to verify that safety is barrier-, not layout-based.
  // CHECK-LABEL: tt.func public @fp8_bk128_256x128_candidate
  // CHECK: } {hcu.block_pingpong.buffer_load_stage3
  // INTEGRATED-LABEL: tt.func public @fp8_bk128_256x128_candidate
  // INTEGRATED: %[[SINGLE_A:[a-zA-Z0-9_]+]] = ttg.local_alloc : () -> !ttg.memdesc<256x128xf8E4M3FN
  // INTEGRATED-NOT: ttg.local_alloc {{.*}}!ttg.memdesc<256x128xf8E4M3FN
  // INTEGRATED-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<128x128xf8E4M3FN
  // Both A halves are read before a full-CTA local barrier permits M1 to
  // overwrite the physical A slot. B remains double buffered.
  // INTEGRATED: scf.for
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<256x64xf8E4M3FN
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<64x128xf8E4M3FN
  // INTEGRATED: tt.load {{.*}}tensor<256x128x!tt.ptr<f8E4M3FN>
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<256x64xf8E4M3FN
  // INTEGRATED: ttg.local_barrier
  // INTEGRATED: tt.load {{.*}}tensor<128x128x!tt.ptr<f8E4M3FN>
  // INTEGRATED: tt.dot {{.*}}tensor<256x64xf8E4M3FN
  // INTEGRATED: ttg.local_barrier
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<64x128xf8E4M3FN
  // INTEGRATED: ttg.local_store {{.*}} -> !ttg.memdesc<128x128xf8E4M3FN
  // INTEGRATED: ttg.local_store {{.*}}, %[[SINGLE_A]]
  // INTEGRATED: gpu.barrier
  // INTEGRATED: tt.dot {{.*}}tensor<256x64xf8E4M3FN
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  // The joined epilogue needs the same cross-wave read-before-overwrite
  // protection for the joined epilogue's single physical A slot.
  // INTEGRATED: amdg.cond_barrier
  // INTEGRATED: ttg.local_load {{.*}} !ttg.memdesc<256x128xf8E4M3FN
  // INTEGRATED-NEXT: ttg.local_barrier
  // INTEGRATED-NEXT: ttg.local_store {{.*}}, %[[SINGLE_A]]
  // INTEGRATED-NEXT: ttg.local_barrier
  tt.func public @fp8_bk128_256x128_candidate(
      %a_ptrs: tensor<256x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8_bk128> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8> {tt.contiguity = dense<[16, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 128 : i32})
      -> tensor<256x128xf32, #mma_fp8_legacy> {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma_fp8_legacy>
    %result = scf.for %iv = %c0_i32 to %K step %c128_i32
        iter_args(%acc = %zero) -> (tensor<256x128xf32, #mma_fp8_legacy>) : i32 {
      %a = tt.load %a_ptrs : tensor<256x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8_bk128>
      %b = tt.load %b_ptrs : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8>
      %dot_a = ttg.convert_layout %a : tensor<256x128xf8E4M3FN, #blocked_a_fp8_bk128> -> tensor<256x128xf8E4M3FN, #dot_a_fp8_legacy>
      %dot_b = ttg.convert_layout %b : tensor<128x128xf8E4M3FN, #blocked_b_fp8> -> tensor<128x128xf8E4M3FN, #dot_b_fp8_legacy>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<256x128xf8E4M3FN, #dot_a_fp8_legacy> * tensor<128x128xf8E4M3FN, #dot_b_fp8_legacy>
          -> tensor<256x128xf32, #mma_fp8_legacy>
      scf.yield %next : tensor<256x128xf32, #mma_fp8_legacy>
    }
    tt.return %result : tensor<256x128xf32, #mma_fp8_legacy>
  }

  // Single-buffer safety is defined by the full-CTA read-before-overwrite
  // barriers, not a particular threadsPerWarp decomposition. Verify that an
  // alternative legal producer layout remains eligible.
  // CHECK-LABEL: tt.func public @fp8_bk128_alternative_a_groups
  // CHECK: } {hcu.block_pingpong.buffer_load_stage3
  // INTEGRATED-LABEL: tt.func public @fp8_bk128_alternative_a_groups
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  tt.func public @fp8_bk128_alternative_a_groups(
      %a_ptrs: tensor<256x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8> {tt.contiguity = dense<[16, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 128 : i32})
      -> tensor<256x128xf32, #mma_fp8> {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma_fp8>
    %result = scf.for %iv = %c0_i32 to %K step %c128_i32
        iter_args(%acc = %zero) -> (tensor<256x128xf32, #mma_fp8>) : i32 {
      %a = tt.load %a_ptrs : tensor<256x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8>
      %b = tt.load %b_ptrs : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8>
      %dot_a = ttg.convert_layout %a : tensor<256x128xf8E4M3FN, #blocked_a_fp8> -> tensor<256x128xf8E4M3FN, #dot_a_fp8>
      %dot_b = ttg.convert_layout %b : tensor<128x128xf8E4M3FN, #blocked_b_fp8> -> tensor<128x128xf8E4M3FN, #dot_b_fp8>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<256x128xf8E4M3FN, #dot_a_fp8> * tensor<128x128xf8E4M3FN, #dot_b_fp8>
          -> tensor<256x128xf32, #mma_fp8>
      scf.yield %next : tensor<256x128xf32, #mma_fp8>
    }
    tt.return %result : tensor<256x128xf32, #mma_fp8>
  }

  // The physical phase split is warp0-3/warp4-7, placing one wave from each
  // phase on every SIMD.  Both phases consume the complete B tile, so full-CTA
  // local barriers protect its single physical 32 KiB slot.  Use the gfx936
  // LEGACY MMAC spelling to verify that this contract is layout-independent.
  // Two physical A slots plus that B slot use exactly 64 KiB.
  // CHECK-LABEL: tt.func public @fp8_bk128_128x256_single_b_candidate
  // CHECK: } {hcu.block_pingpong.buffer_load_stage3
  // INTEGRATED-LABEL: tt.func public @fp8_bk128_128x256_single_b_candidate
  // INTEGRATED-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<128x128xf8E4M3FN
  // INTEGRATED: %[[SINGLE_B:[a-zA-Z0-9_]+]] = ttg.local_alloc : () -> !ttg.memdesc<128x256xf8E4M3FN
  // INTEGRATED-NOT: ttg.local_alloc {{.*}}!ttg.memdesc<128x256xf8E4M3FN
  // INTEGRATED: %[[LOW:[a-zA-Z0-9_]+]] = arith.cmpi ult, {{.*}}, %c4_i32
  // INTEGRATED: %[[HIGH:[a-zA-Z0-9_]+]] = arith.cmpi uge, {{.*}}, %c4_i32
  // INTEGRATED: amdg.cond_barrier %[[LOW]]
  // INTEGRATED: scf.for
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<64x256xf8E4M3FN
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<128x64xf8E4M3FN
  // INTEGRATED: tt.load {{.*}}tensor<128x256x!tt.ptr<f8E4M3FN>
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<64x256xf8E4M3FN
  // INTEGRATED: rocdl.s.barrier
  // INTEGRATED: tt.load {{.*}}tensor<128x128x!tt.ptr<f8E4M3FN>
  // INTEGRATED: tt.dot {{.*}}tensor<128x64xf8E4M3FN
  // INTEGRATED: ttg.local_barrier
  // INTEGRATED: ttg.local_store {{.*}}tensor<128x128xf8E4M3FN
  // INTEGRATED: scf.if %[[LOW]]
  // INTEGRATED: ttg.local_store {{.*}}, %[[SINGLE_B]]
  // INTEGRATED: ttg.local_load {{.*}}tensor<128x64xf8E4M3FN
  // INTEGRATED: gpu.barrier
  // INTEGRATED: tt.dot {{.*}} tensor<128x64xf8E4M3FN
  // INTEGRATED: scf.if %[[HIGH]]
  // INTEGRATED: ttg.local_store {{.*}}, %[[SINGLE_B]]
  // INTEGRATED: ttg.local_barrier
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  // INTEGRATED: amdg.cond_barrier %[[HIGH]]
  // The joined epilogue first retires all reads of the old B generation, then
  // publishes all writes of the new B generation before its cross-wave load.
  // INTEGRATED: ttg.local_load {{.*}} : !ttg.memdesc<128x256xf8E4M3FN
  // INTEGRATED-NEXT: ttg.local_barrier
  // INTEGRATED-NEXT: ttg.local_store {{.*}}, %[[SINGLE_B]]
  // INTEGRATED-NEXT: ttg.local_barrier
  tt.func public @fp8_bk128_128x256_single_b_candidate(
      %a_ptrs: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<128x256x!tt.ptr<f8E4M3FN>, #blocked_b_fp8_bk128> {tt.contiguity = dense<[16, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 128 : i32}) -> tensor<128x256xf32, #mma_fp8_n_legacy> {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma_fp8_n_legacy>
    %result = scf.for %iv = %c0_i32 to %K step %c128_i32
        iter_args(%acc = %zero) -> (tensor<128x256xf32, #mma_fp8_n_legacy>) : i32 {
      %a = tt.load %a_ptrs : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8>
      %b = tt.load %b_ptrs : tensor<128x256x!tt.ptr<f8E4M3FN>, #blocked_b_fp8_bk128>
      %dot_a = ttg.convert_layout %a : tensor<128x128xf8E4M3FN, #blocked_a_fp8> -> tensor<128x128xf8E4M3FN, #dot_a_fp8_n_legacy>
      %dot_b = ttg.convert_layout %b : tensor<128x256xf8E4M3FN, #blocked_b_fp8_bk128> -> tensor<128x256xf8E4M3FN, #dot_b_fp8_n_legacy>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<128x128xf8E4M3FN, #dot_a_fp8_n_legacy> * tensor<128x256xf8E4M3FN, #dot_b_fp8_n_legacy>
          -> tensor<128x256xf32, #mma_fp8_n_legacy>
      scf.yield %next : tensor<128x256xf32, #mma_fp8_n_legacy>
    }
    tt.return %result : tensor<128x256xf32, #mma_fp8_n_legacy>
  }

  // Bit16 BK64 has the same physical 32 KiB A / 16 KiB B geometry as Bit8
  // BK128.  Alias A only and retain the separately selected B generations.
  // CHECK-LABEL: tt.func public @bf16_bk64_256x128_single_a_candidate
  // CHECK: } {hcu.block_pingpong.buffer_load_stage3
  // INTEGRATED-LABEL: tt.func public @bf16_bk64_256x128_single_a_candidate
  // INTEGRATED: %[[BF16_SINGLE_A:[a-zA-Z0-9_]+]] = ttg.local_alloc : () -> !ttg.memdesc<256x64xbf16
  // INTEGRATED-NOT: ttg.local_alloc {{.*}}!ttg.memdesc<256x64xbf16
  // INTEGRATED-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16
  // INTEGRATED: scf.for
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<256x32xbf16
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<32x128xbf16
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<256x32xbf16
  // INTEGRATED: ttg.local_barrier
  // INTEGRATED: tt.dot {{.*}}tensor<256x32xbf16
  // INTEGRATED: ttg.local_barrier
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<32x128xbf16
  // INTEGRATED: ttg.local_store {{.*}} -> !ttg.memdesc<64x128xbf16
  // INTEGRATED: ttg.local_store {{.*}}, %[[BF16_SINGLE_A]]
  // INTEGRATED: gpu.barrier
  // INTEGRATED: tt.dot {{.*}}tensor<256x32xbf16
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  tt.func public @bf16_bk64_256x128_single_a_candidate(
      %a_ptrs: tensor<256x64x!tt.ptr<bf16>, #blocked_a_bf16_bk64> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<64x128x!tt.ptr<bf16>, #blocked_b_bf16_bk64> {tt.contiguity = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 64 : i32})
      -> tensor<256x128xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %result = scf.for %iv = %c0_i32 to %K step %c64_i32
        iter_args(%acc = %zero) -> (tensor<256x128xf32, #mma>) : i32 {
      %a = tt.load %a_ptrs : tensor<256x64x!tt.ptr<bf16>, #blocked_a_bf16_bk64>
      %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<bf16>, #blocked_b_bf16_bk64>
      %dot_a = ttg.convert_layout %a : tensor<256x64xbf16, #blocked_a_bf16_bk64> -> tensor<256x64xbf16, #dot_a>
      %dot_b = ttg.convert_layout %b : tensor<64x128xbf16, #blocked_b_bf16_bk64> -> tensor<64x128xbf16, #dot_b>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<256x64xbf16, #dot_a> * tensor<64x128xbf16, #dot_b>
          -> tensor<256x128xf32, #mma>
      scf.yield %next : tensor<256x128xf32, #mma>
    }
    tt.return %result : tensor<256x128xf32, #mma>
  }

  // The symmetric Bit16 extension aliases the 32 KiB B operand and retains
  // two 16 KiB A generations. Its producer and MFMA ownership are transposed
  // independently from the single-A path above.
  // CHECK-LABEL: tt.func public @bf16_bk64_128x256_single_b_candidate
  // CHECK: } {hcu.block_pingpong.buffer_load_stage3
  // INTEGRATED-LABEL: tt.func public @bf16_bk64_128x256_single_b_candidate
  // INTEGRATED-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16
  // INTEGRATED: %[[BF16_SINGLE_B:[a-zA-Z0-9_]+]] = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16
  // INTEGRATED-NOT: ttg.local_alloc {{.*}}!ttg.memdesc<64x256xbf16
  // INTEGRATED: scf.for
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<32x256xbf16
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<128x32xbf16
  // INTEGRATED: ttg.local_load {{.*}} -> tensor<32x256xbf16
  // INTEGRATED: rocdl.s.barrier
  // INTEGRATED: tt.dot {{.*}}tensor<128x32xbf16
  // INTEGRATED: ttg.local_barrier
  // INTEGRATED: ttg.local_store {{.*}}tensor<128x64xbf16
  // INTEGRATED: ttg.local_store {{.*}}, %[[BF16_SINGLE_B]]
  // INTEGRATED: ttg.local_load {{.*}}tensor<128x32xbf16
  // INTEGRATED: gpu.barrier
  // INTEGRATED: tt.dot {{.*}}tensor<128x32xbf16
  // INTEGRATED: ttg.local_store {{.*}}, %[[BF16_SINGLE_B]]
  // INTEGRATED: ttg.local_barrier
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  tt.func public @bf16_bk64_128x256_single_b_candidate(
      %a_ptrs: tensor<128x64x!tt.ptr<bf16>, #blocked_a_bf16_bk64> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<64x256x!tt.ptr<bf16>, #blocked_b_bf16_bk64> {tt.contiguity = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 64 : i32})
      -> tensor<128x256xf32, #mma_n> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma_n>
    %result = scf.for %iv = %c0_i32 to %K step %c64_i32
        iter_args(%acc = %zero) -> (tensor<128x256xf32, #mma_n>) : i32 {
      %a = tt.load %a_ptrs : tensor<128x64x!tt.ptr<bf16>, #blocked_a_bf16_bk64>
      %b = tt.load %b_ptrs : tensor<64x256x!tt.ptr<bf16>, #blocked_b_bf16_bk64>
      %dot_a = ttg.convert_layout %a : tensor<128x64xbf16, #blocked_a_bf16_bk64> -> tensor<128x64xbf16, #dot_a_n>
      %dot_b = ttg.convert_layout %b : tensor<64x256xbf16, #blocked_b_bf16_bk64> -> tensor<64x256xbf16, #dot_b_n>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<128x64xbf16, #dot_a_n> * tensor<64x256xbf16, #dot_b_n>
          -> tensor<128x256xf32, #mma_n>
      scf.yield %next : tensor<128x256xf32, #mma_n>
    }
    tt.return %result : tensor<128x256xf32, #mma_n>
  }

  // CHECK-LABEL: tt.func public @buffer_load_candidate
  // PIPELINE-LABEL: tt.func public @buffer_load_candidate
  // PIPELINE: ttg.local_alloc : () -> !ttg.memdesc<2x256x32xf16
  // PIPELINE: ttg.local_alloc : () -> !ttg.memdesc<2x32x256xf16
  tt.func public @buffer_load_candidate(
      %a_ptrs: tensor<256x32x!tt.ptr<f16>, #blocked_a> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<32x256x!tt.ptr<f16>, #blocked_b> {tt.contiguity = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>})
      -> tensor<256x256xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    // CHECK: scf.for
    %result = scf.for %iv = %c0_i32 to %c64_i32 step %c32_i32
        iter_args(%acc = %zero) -> (tensor<256x256xf32, #mma>) : i32 {
      %a = tt.load %a_ptrs : tensor<256x32x!tt.ptr<f16>, #blocked_a>
      %b = tt.load %b_ptrs : tensor<32x256x!tt.ptr<f16>, #blocked_b>
      %dot_a = ttg.convert_layout %a : tensor<256x32xf16, #blocked_a> -> tensor<256x32xf16, #dot_a>
      %dot_b = ttg.convert_layout %b : tensor<32x256xf16, #blocked_b> -> tensor<32x256xf16, #dot_b>
      // CHECK: tt.dot
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<256x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
          -> tensor<256x256xf32, #mma>
      scf.yield %next : tensor<256x256xf32, #mma>
    }
    // CHECK: } {hcu.block_pingpong.buffer_load_stage3
    // CHECK-SAME: tt.num_stages = 3 : i32
    // CHECK-SAME: ttg.pipeline.hcu_local_store_stage = 1 : i32
    // CHECK-SAME: ttg.pipeline.hcu_num_buffers = 2 : i32
    tt.return %result : tensor<256x256xf32, #mma>
  }

  // BM and BN must both be at least 128. An asymmetric tile with one larger
  // dimension is eligible when its BK halves are legal and both global loads
  // remain 128-bit.  An auxiliary scale load that is not part of either dot
  // operand stays in the register pipeline and does not hide the A/B chains.
  // CHECK-LABEL: tt.func public @asymmetric_128x256_candidate
  // INTEGRATED-LABEL: tt.func public @asymmetric_128x256_candidate
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  tt.func public @asymmetric_128x256_candidate(
      %a_ptrs: tensor<128x32x!tt.ptr<f16>, #blocked_a> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<32x256x!tt.ptr<f16>, #blocked_b> {tt.contiguity = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %scale_ptr: !tt.ptr<f32>,
      %K: i32 {tt.divisibility = 32 : i32}) -> tensor<128x256xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %result = scf.for %iv = %c0_i32 to %K step %c32_i32
        iter_args(%acc = %zero) -> (tensor<128x256xf32, #mma>) : i32 {
      %a = tt.load %a_ptrs : tensor<128x32x!tt.ptr<f16>, #blocked_a>
      %b = tt.load %b_ptrs : tensor<32x256x!tt.ptr<f16>, #blocked_b>
      %dot_a = ttg.convert_layout %a : tensor<128x32xf16, #blocked_a> -> tensor<128x32xf16, #dot_a>
      %dot_b = ttg.convert_layout %b : tensor<32x256xf16, #blocked_b> -> tensor<32x256xf16, #dot_b>
      %scale = tt.load %scale_ptr : !tt.ptr<f32>
      %scale_tile = tt.splat %scale : f32 -> tensor<128x256xf32, #mma>
      // CHECK: tt.dot
      %tile = tt.dot %dot_a, %dot_b, %zero, inputPrecision = tf32
          : tensor<128x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
          -> tensor<128x256xf32, #mma>
      %scaled = arith.mulf %tile, %scale_tile : tensor<128x256xf32, #mma>
      %next = arith.addf %acc, %scaled : tensor<128x256xf32, #mma>
      scf.yield %next : tensor<128x256xf32, #mma>
    }
    // CHECK: } {hcu.block_pingpong.buffer_load_stage3
    // CHECK-SAME: tt.num_stages = 3 : i32
    tt.return %result : tensor<128x256xf32, #mma>
  }

  // MoE kernels mask padded token rows, but that mask is computed outside the
  // K loop and does not describe a partial K tile. Keep it on the pipelined A
  // load instead of routing the loop through the K-tail splitter.
  // CHECK-LABEL: tt.func public @loop_invariant_token_mask
  // PIPELINE-LABEL: tt.func public @loop_invariant_token_mask
  // PIPELINE: ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16
  // PIPELINE: ttg.local_alloc : () -> !ttg.memdesc<2x32x256xf16
  tt.func public @loop_invariant_token_mask(
      %a_ptrs: tensor<128x32x!tt.ptr<f16>, #blocked_a> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<32x256x!tt.ptr<f16>, #blocked_b> {tt.contiguity = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %token_mask: tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked_a}>>)
      -> tensor<128x256xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %token_mask_2d = tt.expand_dims %token_mask {axis = 1 : i32}
        : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked_a}>>
          -> tensor<128x1xi1, #blocked_a>
    // CHECK: %[[LOAD_MASK:.*]] = tt.broadcast
    // CHECK-SAME: tt.constancy = dense<[1, 32]>
    %load_mask = tt.broadcast %token_mask_2d
        : tensor<128x1xi1, #blocked_a> -> tensor<128x32xi1, #blocked_a>
    %result = scf.for %iv = %c0_i32 to %c64_i32 step %c32_i32
        iter_args(%acc = %zero) -> (tensor<128x256xf32, #mma>) : i32 {
      %a = tt.load %a_ptrs, %load_mask
          : tensor<128x32x!tt.ptr<f16>, #blocked_a>
      %b = tt.load %b_ptrs : tensor<32x256x!tt.ptr<f16>, #blocked_b>
      %dot_a = ttg.convert_layout %a
          : tensor<128x32xf16, #blocked_a> -> tensor<128x32xf16, #dot_a>
      %dot_b = ttg.convert_layout %b
          : tensor<32x256xf16, #blocked_b> -> tensor<32x256xf16, #dot_b>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<128x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
          -> tensor<128x256xf32, #mma>
      scf.yield %next : tensor<128x256xf32, #mma>
    }
    // CHECK: } {hcu.block_pingpong.buffer_load_stage3
    // CHECK-SAME: tt.num_stages = 3 : i32
    tt.return %result : tensor<128x256xf32, #mma>
  }

  // CHECK-LABEL: tt.func public @candidate_128x128_bk32
  // INTEGRATED-LABEL: tt.func public @candidate_128x128_bk32
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  tt.func public @candidate_128x128_bk32(
      %a_ptrs: tensor<128x32x!tt.ptr<f16>, #blocked_a> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<32x128x!tt.ptr<f16>, #blocked_b> {tt.contiguity = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 32 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %result = scf.for %iv = %c0_i32 to %K step %c32_i32
        iter_args(%acc = %zero) -> (tensor<128x128xf32, #mma>) : i32 {
      %a = tt.load %a_ptrs : tensor<128x32x!tt.ptr<f16>, #blocked_a>
      %b = tt.load %b_ptrs : tensor<32x128x!tt.ptr<f16>, #blocked_b>
      %dot_a = ttg.convert_layout %a : tensor<128x32xf16, #blocked_a> -> tensor<128x32xf16, #dot_a>
      %dot_b = ttg.convert_layout %b : tensor<32x128xf16, #blocked_b> -> tensor<32x128xf16, #dot_b>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<128x32xf16, #dot_a> * tensor<32x128xf16, #dot_b>
          -> tensor<128x128xf32, #mma>
      scf.yield %next : tensor<128x128xf32, #mma>
    }
    // CHECK: } {hcu.block_pingpong.buffer_load_stage3
    // CHECK-SAME: tt.num_stages = 3 : i32
    tt.return
  }

  // Bit16 BK64 also fits exactly in 64 KiB of double-buffered LDS for a
  // 128x128 output tile, and each K32 half remains a complete MMAC tile.
  // CHECK-LABEL: tt.func public @candidate_128x128_bk64
  // INTEGRATED-LABEL: tt.func public @candidate_128x128_bk64
  // INTEGRATED: } {hcu.block_pingpong.applied,
  // INTEGRATED-SAME: hcu.block_pingpong.static_buffers
  tt.func public @candidate_128x128_bk64(
      %a_ptrs: tensor<128x64x!tt.ptr<f16>, #blocked_a> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<64x128x!tt.ptr<f16>, #blocked_b> {tt.contiguity = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 64 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %result = scf.for %iv = %c0_i32 to %K step %c64_i32
        iter_args(%acc = %zero) -> (tensor<128x128xf32, #mma>) : i32 {
      %a = tt.load %a_ptrs : tensor<128x64x!tt.ptr<f16>, #blocked_a>
      %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<f16>, #blocked_b>
      %dot_a = ttg.convert_layout %a : tensor<128x64xf16, #blocked_a> -> tensor<128x64xf16, #dot_a>
      %dot_b = ttg.convert_layout %b : tensor<64x128xf16, #blocked_b> -> tensor<64x128xf16, #dot_b>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<128x64xf16, #dot_a> * tensor<64x128xf16, #dot_b>
          -> tensor<128x128xf32, #mma>
      scf.yield %next : tensor<128x128xf32, #mma>
    }
    // CHECK: } {hcu.block_pingpong.buffer_load_stage3
    // CHECK-SAME: tt.num_stages = 3 : i32
    tt.return
  }

  // A full BK16 dot is legal, but splitting it would create K8 halves that do
  // not cover one complete HCU MFMA K instruction.
  // CHECK-LABEL: tt.func public @unsupported_bk_half
  tt.func public @unsupported_bk_half(
      %a_ptrs: tensor<128x16x!tt.ptr<f16>, #blocked_a> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<16x256x!tt.ptr<f16>, #blocked_b> {tt.contiguity = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    // CHECK: scf.for
    // CHECK-NOT: hcu.block_pingpong
    %result = scf.for %iv = %c0_i32 to %c32_i32 step %c16_i32
        iter_args(%acc = %zero) -> (tensor<128x256xf32, #mma>) : i32 {
      %a = tt.load %a_ptrs : tensor<128x16x!tt.ptr<f16>, #blocked_a>
      %b = tt.load %b_ptrs : tensor<16x256x!tt.ptr<f16>, #blocked_b>
      %dot_a = ttg.convert_layout %a : tensor<128x16xf16, #blocked_a> -> tensor<128x16xf16, #dot_a>
      %dot_b = ttg.convert_layout %b : tensor<16x256xf16, #blocked_b> -> tensor<16x256xf16, #dot_b>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<128x16xf16, #dot_a> * tensor<16x256xf16, #dot_b>
          -> tensor<128x256xf32, #mma>
      scf.yield %next : tensor<128x256xf32, #mma>
    }
    // CHECK: tt.return
    tt.return
  }

  // BlockPingpong carries two future A/B tiles in VGPRs. Scalarized global
  // loads make that live range spill, so a structurally matching loop without
  // proven 128-bit load vectorization must remain on the ordinary pipeline.
  // CHECK-LABEL: tt.func public @insufficient_load_alignment
  tt.func public @insufficient_load_alignment(
      %a_ptrs: tensor<256x32x!tt.ptr<f16>, #blocked_a>,
      %b_ptrs: tensor<32x256x!tt.ptr<f16>, #blocked_b>) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    // CHECK: scf.for
    // CHECK-NOT: hcu.block_pingpong
    %result = scf.for %iv = %c0_i32 to %c64_i32 step %c32_i32
        iter_args(%acc = %zero) -> (tensor<256x256xf32, #mma>) : i32 {
      %a = tt.load %a_ptrs : tensor<256x32x!tt.ptr<f16>, #blocked_a>
      %b = tt.load %b_ptrs : tensor<32x256x!tt.ptr<f16>, #blocked_b>
      %dot_a = ttg.convert_layout %a
          : tensor<256x32xf16, #blocked_a> -> tensor<256x32xf16, #dot_a>
      %dot_b = ttg.convert_layout %b
          : tensor<32x256xf16, #blocked_b> -> tensor<32x256xf16, #dot_b>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<256x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
          -> tensor<256x256xf32, #mma>
      scf.yield %next : tensor<256x256xf32, #mma>
    }
    // CHECK: tt.return
    tt.return
  }

  // Split a K-only masked loop into the conventional unmasked main range
  // followed by a one-shot masked tail. Replacing the original step-aligned
  // IV with main_upper = K - K % BK loses its mathematical BK divisibility
  // unless it is preserved explicitly, which would scalarize the tail load.
  // CHECK-LABEL: tt.func public @masked_k_candidate
  // INTEGRATED-LABEL: tt.func public @masked_k_candidate
  // INTEGRATED-NOT: !ttg.memdesc<2x256x32xf16
  // INTEGRATED: ttg.local_alloc : () -> !ttg.memdesc<256x32xf16
  // INTEGRATED: ttg.local_alloc : () -> !ttg.memdesc<32x256xf16
  // INTEGRATED: scf.for
  // INTEGRATED: } {hcu.block_pingpong.applied, hcu.block_pingpong.static_buffers
  tt.func public @masked_k_candidate(
      %a_ptrs: tensor<256x32x!tt.ptr<f16>, #blocked_a> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<32x256x!tt.ptr<f16>, #blocked_b> {tt.contiguity = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %token_mask: tensor<256xi1, #ttg.slice<{dim = 1, parent = #blocked_a}>>,
      %K: i32 {tt.divisibility = 16 : i32})
      -> tensor<256x256xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %a_step = arith.constant dense<32> : tensor<256x32xi32, #blocked_a>
    %b_step = arith.constant dense<32> : tensor<32x256xi32, #blocked_b>
    %a_range = tt.make_range {start = 0 : i32, end = 32 : i32}
        : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %b_range = tt.make_range {start = 0 : i32, end = 32 : i32}
        : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked_b}>>
    %a_range_2d = tt.expand_dims %a_range {axis = 0 : i32}
        : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
          -> tensor<1x32xi32, #blocked_a>
    %b_range_2d = tt.expand_dims %b_range {axis = 1 : i32}
        : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked_b}>>
          -> tensor<32x1xi32, #blocked_b>
    %K_a = tt.splat %K : i32 -> tensor<1x32xi32, #blocked_a>
    %K_b = tt.splat %K : i32 -> tensor<32x1xi32, #blocked_b>
    %token_mask_2d = tt.expand_dims %token_mask {axis = 1 : i32}
        : tensor<256xi1, #ttg.slice<{dim = 1, parent = #blocked_a}>>
          -> tensor<256x1xi1, #blocked_a>
    // CHECK: %[[TOKEN_MASK:.*]] = tt.broadcast
    // CHECK-SAME: tt.constancy = dense<[1, 32]>
    %token_load_mask = tt.broadcast %token_mask_2d
        : tensor<256x1xi1, #blocked_a> -> tensor<256x32xi1, #blocked_a>
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>

    // CHECK: %[[MAIN_UPPER:.*]] = arith.subi {{.*}} {tt.divisibility = 32 : i32}
    // CHECK: scf.for {{.*}} to %[[MAIN_UPPER]] step %{{.*}}
    // CHECK: tt.load {{.*}}, %[[TOKEN_MASK]]
    // CHECK: tt.dot
    // CHECK: } {hcu.block_pingpong.buffer_load_stage3
    // CHECK: scf.if
    // CHECK: hcu.block_pingpong.masked_tail
    %result:3 = scf.for %iv = %c0_i32 to %K step %c32_i32
        iter_args(%acc = %zero, %a_iter = %a_ptrs, %b_iter = %b_ptrs)
        -> (tensor<256x256xf32, #mma>,
            tensor<256x32x!tt.ptr<f16>, #blocked_a>,
            tensor<32x256x!tt.ptr<f16>, #blocked_b>) : i32 {
      %iv_a = tt.splat %iv : i32 -> tensor<1x32xi32, #blocked_a>
      %iv_b = tt.splat %iv : i32 -> tensor<32x1xi32, #blocked_b>
      %a_index = arith.addi %a_range_2d, %iv_a : tensor<1x32xi32, #blocked_a>
      %b_index = arith.addi %b_range_2d, %iv_b : tensor<32x1xi32, #blocked_b>
      %a_mask_1d = arith.cmpi slt, %a_index, %K_a
          : tensor<1x32xi32, #blocked_a>
      %b_mask_1d = arith.cmpi slt, %b_index, %K_b
          : tensor<32x1xi32, #blocked_b>
      %a_k_mask = tt.broadcast %a_mask_1d
          : tensor<1x32xi1, #blocked_a> -> tensor<256x32xi1, #blocked_a>
      %a_mask = arith.andi %token_load_mask, %a_k_mask
          : tensor<256x32xi1, #blocked_a>
      %b_mask = tt.broadcast %b_mask_1d
          : tensor<32x1xi1, #blocked_b> -> tensor<32x256xi1, #blocked_b>
      %a = tt.load %a_iter, %a_mask
          : tensor<256x32x!tt.ptr<f16>, #blocked_a>
      %b = tt.load %b_iter, %b_mask
          : tensor<32x256x!tt.ptr<f16>, #blocked_b>
      %dot_a = ttg.convert_layout %a
          : tensor<256x32xf16, #blocked_a> -> tensor<256x32xf16, #dot_a>
      %dot_b = ttg.convert_layout %b
          : tensor<32x256xf16, #blocked_b> -> tensor<32x256xf16, #dot_b>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<256x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
          -> tensor<256x256xf32, #mma>
      %a_next = tt.addptr %a_iter, %a_step
          : tensor<256x32x!tt.ptr<f16>, #blocked_a>,
            tensor<256x32xi32, #blocked_a>
      %b_next = tt.addptr %b_iter, %b_step
          : tensor<32x256x!tt.ptr<f16>, #blocked_b>,
            tensor<32x256xi32, #blocked_b>
      scf.yield %next, %a_next, %b_next
          : tensor<256x256xf32, #mma>,
            tensor<256x32x!tt.ptr<f16>, #blocked_a>,
            tensor<32x256x!tt.ptr<f16>, #blocked_b>
    }
    tt.return %result#0 : tensor<256x256xf32, #mma>
  }

  // A Bit8 BK128 loop with a K-tail remains on the ordinary pipeline because
  // the one-shot tail has no LDS chain on which to apply the BK/2 split.
  // CHECK-LABEL: tt.func public @masked_k_bit8_bk128_candidate
  // CHECK-NOT: hcu.block_pingpong
  // INTEGRATED-LABEL: tt.func public @masked_k_bit8_bk128_candidate
  // INTEGRATED-NOT: hcu.block_pingpong
  tt.func public @masked_k_bit8_bk128_candidate(
      %a_ptrs: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8_bk128> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %b_ptrs: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8_bk128> {tt.contiguity = dense<[16, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %K: i32 {tt.divisibility = 16 : i32})
      -> tensor<128x128xf32, #mma_fp8> {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %a_step = arith.constant dense<128> : tensor<128x128xi32, #blocked_a_fp8_bk128>
    %b_step = arith.constant dense<128> : tensor<128x128xi32, #blocked_b_fp8_bk128>
    %a_range = tt.make_range {start = 0 : i32, end = 128 : i32}
        : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_a_fp8_bk128}>>
    %b_range = tt.make_range {start = 0 : i32, end = 128 : i32}
        : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked_b_fp8_bk128}>>
    %a_range_2d = tt.expand_dims %a_range {axis = 0 : i32}
        : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_a_fp8_bk128}>>
          -> tensor<1x128xi32, #blocked_a_fp8_bk128>
    %b_range_2d = tt.expand_dims %b_range {axis = 1 : i32}
        : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked_b_fp8_bk128}>>
          -> tensor<128x1xi32, #blocked_b_fp8_bk128>
    %K_a = tt.splat %K : i32 -> tensor<1x128xi32, #blocked_a_fp8_bk128>
    %K_b = tt.splat %K : i32 -> tensor<128x1xi32, #blocked_b_fp8_bk128>
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_fp8>
    %result:3 = scf.for %iv = %c0_i32 to %K step %c128_i32
        iter_args(%acc = %zero, %a_iter = %a_ptrs, %b_iter = %b_ptrs)
        -> (tensor<128x128xf32, #mma_fp8>,
            tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8_bk128>,
            tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8_bk128>) : i32 {
      %iv_a = tt.splat %iv : i32 -> tensor<1x128xi32, #blocked_a_fp8_bk128>
      %iv_b = tt.splat %iv : i32 -> tensor<128x1xi32, #blocked_b_fp8_bk128>
      %a_index = arith.addi %a_range_2d, %iv_a
          : tensor<1x128xi32, #blocked_a_fp8_bk128>
      %b_index = arith.addi %b_range_2d, %iv_b
          : tensor<128x1xi32, #blocked_b_fp8_bk128>
      %a_mask_1d = arith.cmpi slt, %a_index, %K_a
          : tensor<1x128xi32, #blocked_a_fp8_bk128>
      %b_mask_1d = arith.cmpi slt, %b_index, %K_b
          : tensor<128x1xi32, #blocked_b_fp8_bk128>
      %a_mask = tt.broadcast %a_mask_1d
          : tensor<1x128xi1, #blocked_a_fp8_bk128>
          -> tensor<128x128xi1, #blocked_a_fp8_bk128>
      %b_mask = tt.broadcast %b_mask_1d
          : tensor<128x1xi1, #blocked_b_fp8_bk128>
          -> tensor<128x128xi1, #blocked_b_fp8_bk128>
      %a = tt.load %a_iter, %a_mask
          : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8_bk128>
      %b = tt.load %b_iter, %b_mask
          : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8_bk128>
      %dot_a = ttg.convert_layout %a
          : tensor<128x128xf8E4M3FN, #blocked_a_fp8_bk128>
          -> tensor<128x128xf8E4M3FN, #dot_a_fp8>
      %dot_b = ttg.convert_layout %b
          : tensor<128x128xf8E4M3FN, #blocked_b_fp8_bk128>
          -> tensor<128x128xf8E4M3FN, #dot_b_fp8>
      %next = tt.dot %dot_a, %dot_b, %acc, inputPrecision = tf32
          : tensor<128x128xf8E4M3FN, #dot_a_fp8> * tensor<128x128xf8E4M3FN, #dot_b_fp8>
          -> tensor<128x128xf32, #mma_fp8>
      %a_next = tt.addptr %a_iter, %a_step
          : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8_bk128>,
            tensor<128x128xi32, #blocked_a_fp8_bk128>
      %b_next = tt.addptr %b_iter, %b_step
          : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8_bk128>,
            tensor<128x128xi32, #blocked_b_fp8_bk128>
      scf.yield %next, %a_next, %b_next
          : tensor<128x128xf32, #mma_fp8>,
            tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_a_fp8_bk128>,
            tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked_b_fp8_bk128>
    }
    tt.return %result#0 : tensor<128x128xf32, #mma_fp8>
  }

  // A differently shaped loop is intentionally outside the dedicated HCU
  // schedule. The strict matcher must leave it untouched rather than applying
  // a partially valid BlockPingpong transformation.
  // CHECK-LABEL: tt.func public @unsupported_shape
  tt.func public @unsupported_shape(
      %a_ptrs: tensor<128x32x!tt.ptr<f16>, #blocked_a>,
      %b_ptrs: tensor<32x256x!tt.ptr<f16>, #blocked_b>) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    // CHECK: scf.for
    // CHECK-NOT: hcu.block_pingpong
    %result = scf.for %iv = %c0_i32 to %c64_i32 step %c32_i32
        iter_args(%acc = %zero) -> (tensor<256x256xf32, #mma>) : i32 {
      %a = tt.load %a_ptrs : tensor<128x32x!tt.ptr<f16>, #blocked_a>
      %b = tt.load %b_ptrs : tensor<32x256x!tt.ptr<f16>, #blocked_b>
      scf.yield %acc : tensor<256x256xf32, #mma>
    }
    // CHECK: tt.return
    tt.return
  }
}

// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx936' | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32} {
  // The late HCU pass has independently proved that the future LDS reads
  // may outlive publication. Do not let the common barrier pattern add DS=0.
  // CHECK-LABEL: llvm.func @deferred
  // CHECK: rocdl.wait.asyncmark
  // CHECK-NOT: rocdl.s.waitcnt
  // CHECK: rocdl.s.barrier
  tt.func public @deferred() {
    %token = ttg.async_wait {num = 3 : i32}
    ttg.local_barrier {hcu.block_pingpong.drop_lgkmcnt}
    tt.return
  }

  // Unproved publication retains the normal LDS drain, independently of
  // whether the copy wait allows zero or a nonzero number of groups.
  // CHECK-LABEL: llvm.func @conservative
  // CHECK: rocdl.wait.asyncmark
  // CHECK: rocdl.s.waitcnt
  // CHECK: rocdl.s.barrier
  tt.func public @conservative() {
    %token = ttg.async_wait {num = 2 : i32}
    ttg.local_barrier
    tt.return
  }
}

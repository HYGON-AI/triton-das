// RUN: triton-opt --split-input-file %s --verify-diagnostics

// The common verifier leaves backend-specific initial wave limits to backends.
module attributes {"ttg.num-warps" = 2 : i32} {
  tt.func @two_waves() {
    ttg.warp_specialize()
    default { ttg.warp_yield }
    partition0() num_warps(2) { ttg.warp_return }
    partition1() num_warps(2) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @four_waves() {
    ttg.warp_specialize()
    default { ttg.warp_yield }
    partition0() num_warps(4) { ttg.warp_return }
    partition1() num_warps(4) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @one_wave() {
    ttg.warp_specialize()
    default { ttg.warp_yield }
    partition0() num_warps(2) { ttg.warp_return }
    partition1() num_warps(2) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

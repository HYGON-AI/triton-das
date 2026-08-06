"""HCU regression: abarrier/ebarrier PC .mlir fixtures -> LLIR lowering."""

import os
import re

TRITON_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
HCU_INPUTS = os.path.join(TRITON_ROOT, "test/TritonGPU/hcu/Inputs")


def _read(name: str) -> str:
    with open(os.path.join(HCU_INPUTS, name)) as f:
        return f.read()


def test_hcu_abarrier_pc_llir():
    llir = _read("hcu-abarrier-pc.llir")

    # Kernel signature and thread-block size (4 warps * 64 lanes = 256).
    assert '"nvvm.reqntid"="256"' in llir

    # Partition split: wave-id switch dispatches waves 0..3 and 4..7.
    assert "call i32 (...) @llvm.hcu.get.wave.id()" in llir
    assert re.search(r"switch\s+i32\s+%\d+", llir)
    for wid in range(8):
        assert re.search(rf"i32\s+{wid},\s+label\s+%\d+", llir)

    # Only the first partition emits the epilogue marker.
    assert 'call void asm sideeffect "; mma_epilogue_partition_0", ""()' in llir

    # abarrier intrinsics: init(id, expected=4), arrive(id, cnt=1), try.wait, inv.
    assert re.search(
        r"call void @llvm\.hcu\.s\.abarrier\.init\(i32\s+\d+,\s*i32\s+4\)", llir)
    assert re.search(
        r"call i32 \(i32, \.\.\.\) @llvm\.hcu\.s\.abarrier\.arrive\(i32\s+\d+,\s*i32\s+1\)",
        llir)
    assert "@llvm.hcu.s.abarrier.try.wait" in llir
    assert "@llvm.hcu.s.abarrier.inv" in llir

    # ebarrier.sync is still used for cross-partition rendezvous around the loop.
    assert "@llvm.hcu.s.ebarrier.sync" in llir


def test_hcu_ebarrier_pc_llir():
    llir = _read("hcu-ebarrier-pc.llir")

    # Partition split: wave-id switch dispatches waves 0..3 and 4..7.
    assert "call i32 (...) @llvm.hcu.get.wave.id()" in llir
    assert re.search(r"switch\s+i32\s+%\d+", llir)
    for wid in range(8):
        assert re.search(rf"i32\s+{wid},\s+label\s+%\d+", llir)

    # Only the first partition emits the epilogue marker.
    assert 'call void asm sideeffect "; mma_epilogue_partition_0", ""()' in llir

    # Cross-partition rendezvous uses wvCnt=8; end-of-partition arrive uses wvCnt=4.
    assert re.search(
        r"call void @llvm\.hcu\.s\.ebarrier\.sync\(i32\s+\d+,\s*i32\s+8\)", llir)
    assert re.search(
        r"call void @llvm\.hcu\.s\.ebarrier\.arrive\(i32\s+\d+,\s*i32\s+4\)", llir)

    # Reduction-style ebarrier variants must be preserved by the lowering.
    assert "@llvm.hcu.s.ebarrier.and" in llir
    assert "@llvm.hcu.s.ebarrier.or" in llir
    assert "@llvm.hcu.s.ebarrier.popc" in llir

    # Pure ebarrier fixture: no abarrier intrinsics should appear.
    assert "@llvm.hcu.s.abarrier" not in llir
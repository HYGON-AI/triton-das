"""Check max_memory_cluster_dwords via pre-generated AMDGCN goldens.

Reads Inputs/hcu-max-mem-cluster-*.amdgcn (compiled offline from
hcu-max-mem-cluster.ll). A larger dword budget should yield longer, fewer
load clusters (cap is budget/4 for dwordx4). "default" has no fn attr and
matches LLVM's built-in budget of 8.

Regenerate (cwd: test/TritonGPU/hcu/Inputs/):

  CLANG=/opt/dtk/aillvm/bin/clang-18
  IN=hcu-max-mem-cluster.ll
  OUT=hcu-max-mem-cluster
  BASE=(-target amdgcn-amd-amdhsa -mcpu=gfx936:xnack-
        -mllvm=-check-valu-data-forward-hazards=0
        -mllvm=-disable-cluster-lds-memops=true
        -mllvm=-hcu-pre-emit-load-store-opt=false
        -mllvm=-support-768-vgprs=true
        -mllvm=-enable-hcu-approx-func-fp-math=true
        -mllvm=-hcu-update-wait-by-reverse-search=true
        -O3)
  $CLANG "${BASE[@]}" "$IN" -S -o "$OUT-default.amdgcn"
  for N in 16 24 32 40 48; do
    sed "s/\"amdgpu-waves-per-eu\"=\"1\"/\"amdgpu-max-memory-cluster-dwords\"=\"$N\" \"amdgpu-waves-per-eu\"=\"1\"/" \
      "$IN" | $CLANG "${BASE[@]}" -x ir - -S -o "$OUT-$N.amdgcn"
  done
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

TRITON_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
HCU_INPUTS = os.path.join(TRITON_ROOT, "test/TritonGPU/hcu/Inputs")

_STEM = "hcu-max-mem-cluster"
# Ordered by effective budget ("default" == LLVM budget 8).
_GOLDENS = (
    ("default", f"{_STEM}-default.amdgcn"),
    ("16", f"{_STEM}-16.amdgcn"),
    ("24", f"{_STEM}-24.amdgcn"),
    ("32", f"{_STEM}-32.amdgcn"),
    ("40", f"{_STEM}-40.amdgcn"),
    ("48", f"{_STEM}-48.amdgcn"),
)

_PAYLOAD_LOAD_RE = re.compile(r"^\s*buffer_load_dwordx4\b")
_LABEL_RE = re.compile(r"^[A-Za-z_.][\w.$]*:$")
# Gap > this many real insns between loads => cluster boundary.
_CLUSTER_BREAK_GAP = 3


def _read(name: str) -> str:
    with open(os.path.join(HCU_INPUTS, name)) as f:
        return f.read()


def _is_real_insn(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith(".") or s.startswith(";"):
        return False
    return not _LABEL_RE.match(s)


@dataclass(frozen=True)
class ClusterSchedule:
    n_loads: int
    n_clusters: int
    max_cluster: int
    cluster_sizes: tuple[int, ...]


def analyze_cluster_schedule(amdgcn: str) -> ClusterSchedule:
    lines = amdgcn.splitlines()
    loads = [i for i, l in enumerate(lines) if _PAYLOAD_LOAD_RE.search(l)]
    assert loads, "no buffer_load_dwordx4 payload loads found in GCN assembly"

    sizes: list[int] = []
    cur = 1
    for a, b in zip(loads, loads[1:]):
        gap = sum(1 for l in lines[a + 1 : b] if _is_real_insn(l))
        if gap > _CLUSTER_BREAK_GAP:
            sizes.append(cur)
            cur = 1
        else:
            cur += 1
    sizes.append(cur)

    return ClusterSchedule(
        n_loads=len(loads),
        n_clusters=len(sizes),
        max_cluster=max(sizes),
        cluster_sizes=tuple(sizes),
    )


def test_max_memory_cluster_dwords_amdgcn():
    asms = [(budget, _read(fn)) for budget, fn in _GOLDENS]
    for budget, asm in asms:
        assert asm.strip(), f"budget {budget}: golden GCN assembly is empty"

    scheds = [(budget, analyze_cluster_schedule(asm)) for budget, asm in asms]
    for budget, sched in scheds:
        print(f"budget {budget:>7}: {sched}")

    n_loads = {budget: s.n_loads for budget, s in scheds}
    assert len(set(n_loads.values())) == 1, f"payload load count changed across budgets: {n_loads}"

    for (lo, lo_asm), (hi, hi_asm) in zip(asms, asms[1:]):
        assert lo_asm != hi_asm, f"budget {lo} vs {hi} GCN assembly identical (knob is a no-op)"

    for (lo, lo_s), (hi, hi_s) in zip(scheds, scheds[1:]):
        assert lo_s.max_cluster < hi_s.max_cluster, (
            f"expected longest load cluster to grow from budget {lo} to {hi}, got "
            f"{lo_s.max_cluster} -> {hi_s.max_cluster}"
        )
        assert lo_s.n_clusters >= hi_s.n_clusters, (
            f"expected no more load clusters from budget {lo} to {hi}, got "
            f"{lo_s.n_clusters} -> {hi_s.n_clusters}"
        )
    assert scheds[0][1].n_clusters > scheds[-1][1].n_clusters, (
        f"expected fewer load clusters at budget {scheds[-1][0]} than at "
        f"{scheds[0][0]}, got {scheds[0][1].n_clusters} -> {scheds[-1][1].n_clusters}"
    )


if __name__ == "__main__":
    test_max_memory_cluster_dwords_amdgcn()
    print("PASS: max_memory_cluster_dwords GCN golden checks")

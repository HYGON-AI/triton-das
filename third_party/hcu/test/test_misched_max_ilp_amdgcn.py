"""Check misched / max-ilp clang knobs via pre-generated AMDGCN goldens.

Reads Inputs/hcu-misched-max-ilp-sched-*.amdgcn (compiled offline from
hcu-misched-max-ilp-sched.ll). Expected aggregation strength:

  max_ilp  >  no_regpressure  >  baseline
  both flags ~= max_ilp (max_ilp dominates)

Knobs:
  -mllvm=-amdgpu-enable-max-ilp-scheduling-strategy=true
  -mllvm=-misched-regpressure=false

Regenerate (cwd: test/TritonGPU/hcu/Inputs/):

  CLANG=/opt/dtk/aillvm/bin/clang-18
  IN=hcu-misched-max-ilp-sched.ll
  OUT=hcu-misched-max-ilp-sched
  BASE=(-target amdgcn-amd-amdhsa -mcpu=gfx936:xnack-
        -mllvm=-check-valu-data-forward-hazards=0
        -mllvm=-disable-cluster-lds-memops=true
        -mllvm=-hcu-pre-emit-load-store-opt=false
        -mllvm=-support-768-vgprs=true
        -mllvm=-enable-hcu-approx-func-fp-math=true
        -mllvm=-hcu-update-wait-by-reverse-search=true
        -O3)
  MAXILP=-mllvm=-amdgpu-enable-max-ilp-scheduling-strategy=true
  NOREG=-mllvm=-misched-regpressure=false
  $CLANG "${BASE[@]}"                "$IN" -S -o "$OUT-baseline.amdgcn"
  $CLANG "${BASE[@]}" $MAXILP        "$IN" -S -o "$OUT-max-ilp.amdgcn"
  $CLANG "${BASE[@]}" $NOREG         "$IN" -S -o "$OUT-no-regpressure.amdgcn"
  $CLANG "${BASE[@]}" $MAXILP $NOREG "$IN" -S -o "$OUT-max-ilp-no-regpressure.amdgcn"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

TRITON_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
HCU_INPUTS = os.path.join(TRITON_ROOT, "test/TritonGPU/hcu/Inputs")

_STEM = "hcu-misched-max-ilp-sched"
_GOLDENS = {
    "baseline": f"{_STEM}-baseline.amdgcn",
    "max_ilp": f"{_STEM}-max-ilp.amdgcn",
    "no_regpressure": f"{_STEM}-no-regpressure.amdgcn",
    "both": f"{_STEM}-max-ilp-no-regpressure.amdgcn",
}

_PAYLOAD_LOAD_RE = re.compile(r"^\s*buffer_load_dwordx4\b")
_WAIT_VMCNT_RE = re.compile(r"s_waitcnt\b.*\bvmcnt\(")
_VALU_RE = re.compile(r"^\s*v_")


def _read(name: str) -> str:
    with open(os.path.join(HCU_INPUTS, name)) as f:
        return f.read()


@dataclass(frozen=True)
class LoadSchedule:
    n_loads: int
    max_cluster: int          # longest load run with no vmcnt wait in between
    waits_between_loads: int  # load-gaps that contain s_waitcnt vmcnt()


def analyze_load_schedule(amdgcn: str) -> LoadSchedule:
    """Group consecutive payload loads; break on vmcnt wait or heavy VALU."""
    lines = amdgcn.splitlines()
    loads = [i for i, l in enumerate(lines) if _PAYLOAD_LOAD_RE.search(l)]
    assert loads, "no buffer_load_dwordx4 payload loads found in GCN assembly"

    clusters: list[int] = []
    cur = 1
    waits = 0
    for a, b in zip(loads, loads[1:]):
        seg = lines[a + 1 : b]
        n_valu = sum(1 for l in seg if _VALU_RE.search(l))
        has_wait = any(_WAIT_VMCNT_RE.search(l) for l in seg)
        if has_wait:
            waits += 1
        if not has_wait and n_valu <= 8:
            cur += 1
        else:
            clusters.append(cur)
            cur = 1
    clusters.append(cur)

    return LoadSchedule(
        n_loads=len(loads),
        max_cluster=max(clusters),
        waits_between_loads=waits,
    )


def test_misched_max_ilp_amdgcn_goldens_effective():
    asms = {name: _read(fn) for name, fn in _GOLDENS.items()}
    for name, asm in asms.items():
        assert asm.strip(), f"{name}: golden GCN assembly is empty"

    names = list(asms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert asms[names[i]] != asms[names[j]], (
                f"{names[i]} and {names[j]} GCN assembly are identical"
            )

    base = analyze_load_schedule(asms["baseline"])
    ilp = analyze_load_schedule(asms["max_ilp"])
    rp = analyze_load_schedule(asms["no_regpressure"])
    both = analyze_load_schedule(asms["both"])
    print(f"baseline:       {base}")
    print(f"max_ilp:        {ilp}")
    print(f"no_regpressure: {rp}")
    print(f"both:           {both}")

    assert base.n_loads == ilp.n_loads == rp.n_loads == both.n_loads, (
        f"payload load count changed: base={base.n_loads} ilp={ilp.n_loads} "
        f"rp={rp.n_loads} both={both.n_loads}"
    )

    assert ilp.max_cluster > rp.max_cluster > base.max_cluster, (
        f"expected load clustering max_ilp > no_regpressure > baseline, got "
        f"ilp={ilp.max_cluster} rp={rp.max_cluster} base={base.max_cluster}"
    )
    assert base.waits_between_loads > rp.waits_between_loads > ilp.waits_between_loads, (
        f"expected vmcnt waits baseline > no_regpressure > max_ilp, got "
        f"base={base.waits_between_loads} rp={rp.waits_between_loads} "
        f"ilp={ilp.waits_between_loads}"
    )

    assert both.max_cluster >= rp.max_cluster, (
        f"both-flags clustered weaker than no_regpressure: "
        f"both={both.max_cluster} rp={rp.max_cluster}"
    )
    assert both.waits_between_loads < rp.waits_between_loads, (
        f"both-flags did not aggregate like max_ilp: "
        f"both waits={both.waits_between_loads} rp waits={rp.waits_between_loads}"
    )


if __name__ == "__main__":
    test_misched_max_ilp_amdgcn_goldens_effective()
    print("PASS: misched/max-ilp GCN golden scheduling checks")

#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Compile matrix-read variants; optionally measure register mappings.

Assembly acceptance is recorded separately from execution and mapping checks.
The output is experimental evidence, not an automatically enabled bank profile.
"""
import argparse
import csv
import io
import json
import pathlib
import subprocess

INSTRUCTIONS = [
    ("ds_read_m32x64_b4", 4), ("ds_read_m32x32_b8", 8),
    ("ds_read_m32x16_b16", 16), ("ds_read_m32x16_b16_alt", 16),
    ("ds_read_m32x8_b32", 32), ("ds_read_m32x32_b8_alt2", 8),
    ("ds_read_m64x16_b8_alt4", 8), ("ds_read_m16x16_b32", 32),
    ("ds_read_mt16x16_b32", 32), ("ds_read_mt16x32_b16_alt2", 16),
]
# These legacy instructions are shared by the HCU targets accepted below. The
# newer instruction variants remain compile-only until a target-specific test
# requests them explicitly.
LEGACY_RUN = {name for name, _ in INSTRUCTIONS[:5]}
EXECUTE_ARCHES = {"gfx92a", "gfx936", "gfx938", "gfx946"}


def decode(output, bits):
    elements = 128 // bits
    mapping = [0] * (64 * elements)
    seen = set()
    for digit, lane, *registers in csv.reader(io.StringIO(output)):
        digit, lane = int(digit), int(lane)
        if not 0 <= lane < 64 or len(registers) != 4 or (digit, lane) in seen:
            raise ValueError("invalid or duplicate lane record")
        seen.add((digit, lane))
        for reg, value in enumerate(map(int, registers)):
            for sub in range(32 // bits):
                item = reg * (32 // bits) + sub
                mapping[lane * elements + item] |= (
                    (value >> (bits * sub)) & ((1 << bits) - 1)) << digit
    if seen != {(d, lane) for d in range(0, 11, bits) for lane in range(64)}:
        raise ValueError("incomplete mapping capture")
    if sorted(mapping) != list(range(8192 // bits)):
        raise ValueError("mapping is not a source-element permutation")
    basis = [mapping[1 << bit] ^ mapping[0]
             for bit in range((len(mapping) - 1).bit_length())]
    for index, expected in enumerate(mapping):
        actual = mapping[0]
        for bit, value in enumerate(basis):
            if index & (1 << bit):
                actual ^= value
        if actual != expected:
            raise ValueError("mapping is not affine over XOR")
    return {"input_order": "element-within-lane bits, then lane bits",
            "origin": mapping[0], "basis": basis, "mapping": mapping}


def main():
    parser = argparse.ArgumentParser(
        description="Compile and inspect HCU LDS matrix-read variants.")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--hipcc", default="/opt/dtk/bin/hipcc")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--instruction", action="append",
                        choices=[name for name, _ in INSTRUCTIONS],
                        help="limit the probe to one or more instructions")
    args = parser.parse_args()
    if args.execute and args.arch not in EXECUTE_ARCHES:
        parser.error("execution is not enabled for the requested HCU target")
    args.output.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path(__file__).with_name("lds_instruction_probe.hip")
    results = []
    selected = set(args.instruction or [name for name, _ in INSTRUCTIONS])
    for name, bits in INSTRUCTIONS:
        if name not in selected:
            continue
        binary = args.output / name
        command = [args.hipcc, f"--offload-arch={args.arch}", "-O2", str(source),
                   f'-DREAD_INSTRUCTION="{name}"', f"-DELEMENT_BITS={bits}",
                   f'-DEXPECT_ARCH="{args.arch}"',
                   "-o", str(binary)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=90)
        (args.output / f"{name}.compile.log").write_text(result.stdout + result.stderr)
        row = {"instruction": name, "arch": args.arch,
               "assembled": result.returncode == 0, "executed": False}
        if result.returncode == 0 and args.execute and name in LEGACY_RUN:
            run = subprocess.run([str(binary)], text=True, capture_output=True, timeout=30)
            (args.output / f"{name}.csv").write_text(run.stdout)
            (args.output / f"{name}.run.log").write_text(run.stderr)
            row["executed"] = run.returncode == 0
            if run.returncode == 0:
                try:
                    row.update(decode(run.stdout, bits))
                    row["mapping_verified"] = True
                except ValueError as error:
                    row["mapping_error"] = str(error)
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print("HCU LDS instruction result recorded", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Collect isolated read2 bank counters. Select an idle GPU externally.

No system installation is modified. Profiles are evidence, not an automatic
authorization to enable a target's phase model.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Collect isolated HCU LDS read bank counters.")
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arch', default='gfx936')
    parser.add_argument('--hipcc', default='/opt/dtk/bin/hipcc')
    parser.add_argument('--xprof', default='/opt/rocm-6.3.3/bin/xprof')
    parser.add_argument('--xcu', default='/opt/xcompute-4.4.4/XCompute/bin/xcu')
    parser.add_argument('--matrix', choices=('ds_read_m32x64_b4',
                                             'ds_read_m32x32_b8',
                                             'ds_read_m32x16_b16',
                                             'ds_read_m32x8_b32'),
                        help='measure uniform-stride matrix reads instead of read2')
    parser.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for bits in ((64,) if args.matrix else (32, 64)):
        for st64 in ((False,) if args.matrix else (False, True)):
            opcode = args.matrix or f'ds_read2{"st64" if st64 else ""}_b{bits}'
            binary = (args.output / opcode).resolve()
            offset = 1 if st64 else 128 // (bits // 8)
            suffix = '' if args.matrix else f'offset0:0 offset1:{offset}'
            subprocess.run([
                args.hipcc, f'--offload-arch={args.arch}', '-O2',
                str(Path(__file__).with_name('lds_bank_probe.hip')),
                f'-DREAD_INSTRUCTION="{opcode}"',
                f'-DREAD_SUFFIX="{suffix}"',
                f'-DRESULT_WORDS={4 if args.matrix else bits // 16}',
                f'-DEXPECT_ARCH="{args.arch}"', '-o', str(binary),
            ], check=True, capture_output=True, timeout=90)
            if args.compile_only:
                print("HCU LDS probe binary compiled")
                continue
            for stride in ((16, 32, 64, 128) if args.matrix else (bits // 8, bits // 4, 128)):
                out = (args.output / f'{opcode}-stride{stride}').resolve()
                # Refuse to mix a previous capture with a new one.
                out.mkdir()
                with (out / 'capture.log').open('w') as log:
                    subprocess.run([
                        args.xprof, '--sections', 'memory_workload',
                        '--kernels', 'lds_bank_probe', '--output-dir', str(out),
                        str(binary), str(stride),
                    ], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                perf, = out.glob('*.perf')
                subprocess.run([args.xcu, 'status', '-P', str(perf), '-S',
                                'memory_workload', '-F', 'csv', '-D', str(out)],
                               check=True, capture_output=True, timeout=120)
                metrics_file, = out.glob('*_Z14lds_bank_probe*.csv')
                with metrics_file.open() as stream:
                    metrics = {r[0]: r[-1] for r in csv.reader(stream) if r}
                row = dict(opcode=opcode, stride=stride,
                           offset1=None if args.matrix else offset,
                           conflicts=metrics['LDS-Bank Conflict'],
                           accesses=metrics['LDS-Index Accesses'], perf=str(perf))
                rows.append(row)
                (args.output / 'results.json').write_text(json.dumps(rows, indent=2))
                print("HCU LDS probe result recorded", flush=True)


if __name__ == '__main__':
    main()

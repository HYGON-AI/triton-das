#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Differential pair-collision experiment; does not assume a phase partition.

Other lanes broadcast bank words 0..3. The selected lanes read banks 4..7
and either 8..11 (control) or different words in 4..7 (collision).
Counter deltas, not result-register routing, are the observation.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(
        description="Measure HCU LDS lane-pair bank conflicts.")
    p.add_argument('--binary', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--seconds', type=int, nargs='+')
    p.add_argument('--partition', action='store_true',
                   help='infer groups by testing each smallest remaining lane against all remaining lanes')
    p.add_argument('--xprof', default='/opt/rocm-6.3.3/bin/xprof')
    p.add_argument('--xcu', default='/opt/xcompute-4.4.4/XCompute/bin/xcu')
    a = p.parse_args()
    if a.partition:
        a.first, a.seconds = 0, list(range(1, 64))
    if not a.seconds or not 0 <= a.first < 64 or any(not 0 <= s < 64 or s == a.first for s in a.seconds):
        p.error('pair must consist of distinct wave64 lanes')
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    groups = []
    remaining = set(range(64))
    group = [a.first]
    for second in a.seconds:
        counters = []
        for collide in (0, 1):
            out = (a.output / f'{a.first}-{second}-{collide}').resolve()
            out.mkdir()
            with (out / 'capture.log').open('w') as log:
                subprocess.run([a.xprof, '--sections', 'memory_workload', '--kernels',
                                'lds_bank_probe', '--output-dir', str(out),
                                a.binary, '16', str(a.first), str(second), str(collide)],
                               check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            perf, = out.glob('*.perf')
            subprocess.run([a.xcu, 'status', '-P', str(perf), '-S', 'memory_workload',
                            '-F', 'csv', '-D', str(out)],
                           check=True, capture_output=True, timeout=120)
            metrics_file, = out.glob('*_Z14lds_bank_probe*.csv')
            with metrics_file.open() as stream:
                metrics = {r[0]: r[-1] for r in csv.reader(stream) if r}
            counters.append(float(metrics['LDS-Bank Conflict']))
        row = dict(first=a.first, second=second, control=counters[0],
                   collision=counters[1], delta=counters[1] - counters[0])
        rows.append(row)
        (a.output / 'results.json').write_text(json.dumps(rows, indent=2))
        print("HCU LDS pair result recorded", flush=True)
        if a.partition:
            if row['control'] != 0 or row['delta'] < 0 or row['delta'] % 1024:
                raise RuntimeError('pair experiment does not fit the assumed conflict/no-conflict model')
            if row['delta']:
                group.append(second)
            if second == max(remaining):
                groups.append(group)
                remaining.difference_update(group)
                (a.output / 'groups.json').write_text(json.dumps(groups, indent=2))
                if remaining:
                    a.first = min(remaining)
                    group = [a.first]
                    # Extend the work list; group members are always removed
                    # before choosing the next representative.
                    a.seconds.extend(sorted(remaining - {a.first}))
                    if len(remaining) == 1:
                        groups.append(group)
                        (a.output / 'groups.json').write_text(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

"""
VGPR Size 自动调试脚本（WASP + WDRA 专用）
根据编译输出自动调整各分支的 VGPR 分配。

WDRA 参数：
  - wdra_num_load_regs
  - wdra_num_mma_regs_main（始终存在）
  - wdra_num_mma_regs_tail（始终传入；仅 1 个 mma 分支时为 0，不参与合法性校验里的“分支”计数）

合法性校验（validate_allocation）只针对硬件 wave 分支（num_branches 路），与 JSON 里占位的 tail=0 无关。
"""

import os
import sys
import re
import multiprocessing
import subprocess
import shutil
import json
import uuid
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional, Set
from itertools import product
from datetime import datetime
from collections import OrderedDict


class WDRATuner:
    def __init__(self, script_path: str, config: Dict[str, int], verbose: bool = False):
        self.script_path = str(Path(script_path).resolve())
        # Use per-config isolated cache dir so multiple configs can run in parallel safely.
        self.cache_dir = Path(tempfile.gettempdir()) / f"triton_wdra_cache_{uuid.uuid4().hex}"
        # Isolated working directory avoids collisions on relative outputs like ./m5out.
        self.work_dir = Path(tempfile.gettempdir()) / f"triton_wdra_work_{uuid.uuid4().hex}"
        self.verbose = verbose
        self.config = dict(config)
        self.config_path = Path(tempfile.gettempdir()) / f"config_{uuid.uuid4().hex}.json"
        self.indent_level = 0
        self.indent_size = 2

        # Hardware Specific
        self.eus_per_cu = 4
        self.total_num_vgpr = 512

        # Kernel Specific（仅考虑 WASP + WDRA 模式）
        self.default_num_warps = 0
        self.load_num_warps = config["wasp_num_load_warps"]
        self.mma_num_warps = config["wasp_num_mma_warps"]
        self.waves_per_tg = self.default_num_warps + self.load_num_warps + self.mma_num_warps
        self.waves_per_eu = int(self.waves_per_tg / self.eus_per_cu)
        self.vgpr_limit = int(self.total_num_vgpr / (self.waves_per_tg / self.eus_per_cu))
        self.num_branches = int(self.waves_per_eu)
        # [4,4] -> 2 分支：1 load + 1 mma；[4,8] -> 3 分支：1 load + 2 mma
        self.num_mma_branches = 1 if self.num_branches == 2 else 2

        self.initial_branch_usage: Tuple[int, ...] = ()
        self.branch_to_partition: Dict[str, Dict[int, str]] = {}
        self.partition_to_branch: Dict[str, Dict[str, list[int]]] = {}

        self.best_alloc_tuple: Tuple[int, ...] | None = None
        self.last_error: str = ""
        self._last_script_output: str = ""
        # 曾实测发生 VGPR spill 的分配索引；减小分配时跳过，避免「减 → 溢出 → 加 → 再减」死循环
        self._alloc_indices_spilled: Set[int] = set()

        _range = range(4, self.vgpr_limit + 1, 4)
        self.all_valid_allocations = sorted(
            [
                alloc
                for alloc in product(_range, repeat=self.num_branches)
                if self.validate_allocation(alloc)
            ],
            key=sum,
        )

        if self.verbose:
            self._log_header("Initialization")
            self._log_info(
                f"Waves: {self.waves_per_tg}/tg, {self.waves_per_eu}/eu | "
                f"Branches: {self.num_branches} | VGPR limit: {self.vgpr_limit}/branch"
            )
            self._log_info(f"Valid allocations: {len(self.all_valid_allocations)}")

    def _output_snippet(self, max_len: int = 400) -> str:
        s = (self._last_script_output or "").replace("\r", " ").replace("\n", " ")
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return s

    def _fail(self, reason: str, with_output: bool = True) -> None:
        self.last_error = reason
        if with_output:
            snip = self._output_snippet()
            if snip:
                self.last_error = f"{reason} | {snip}"

    # -------------------------- logging helpers --------------------------
    def _get_indent(self, level: int = 0) -> str:
        total_level = self.indent_level + level
        return " " * (total_level * self.indent_size)

    def _log(self, message: str, prefix: str, level: int = 0, end: str = "\n"):
        if not self.verbose:
            return
        indent = self._get_indent(level)
        print(f"{indent}{prefix} {message}", end=end)

    def _log_header(self, message: str, level: int = 0):
        if not self.verbose:
            return
        indent = self._get_indent(level)
        width = 70 - len(indent)
        print()
        print(indent + "=" * width)
        print(f"{indent}{message}")
        print(indent + "=" * width)

    def _log_section(self, message: str, level: int = 0):
        if not self.verbose:
            return
        indent = self._get_indent(level)
        width = 70 - len(indent)
        print()
        print(indent + "-" * width)
        print(f"{indent}{message}")
        print(indent + "-" * width)

    def _log_info(self, message: str, level: int = 0):
        self._log(message, "[INFO]", level)

    def _log_success(self, message: str, level: int = 0):
        self._log(message, "[✓ SUCCESS]", level)

    def _log_warning(self, message: str, level: int = 0):
        self._log(message, "[⚠ WARNING]", level)

    def _log_error(self, message: str, level: int = 0):
        self._log(message, "[✗ ERROR]", level)

    def _log_stage(self, message: str, level: int = 0):
        if not self.verbose:
            return
        indent = self._get_indent(level)
        print()
        print(f"{indent}>>> {message}")
        print(f"{indent}{'─' * (len(message) + 4)}")

    def _print_branch_summary(
        self,
        branch_usage: Tuple[int, ...],
        branch_available: Tuple[int, ...],
        mode: str,
        level: int = 1,
    ):
        if not self.verbose:
            return
        mapping = self.branch_to_partition.get(mode, {})
        indent = self._get_indent(level)
        print(f"{indent}Branch Summary:")
        print(
            f"{indent}  {'Branch':<8} {'Usage':<8} {'Available':<10} "
            f"{'MinReq':<8} {'Partition':<10}"
        )
        print(
            f"{indent}  {'─' * 8} {'─' * 8} {'─' * 10} "
            f"{'─' * 8} {'─' * 10}"
        )

        for branch_id in range(self.num_branches):
            usage = branch_usage[branch_id] if branch_id < len(branch_usage) else 0
            available = branch_available[branch_id] if branch_id < len(branch_available) else 0
            min_required = self.round_to_multiple_of_4(usage)
            partition = mapping.get(branch_id, "-") if mapping is not None else "-"
            print(
                f"{indent}  #{branch_id:<7} {usage:<8} {available:<10} "
                f"{min_required:<8} {partition:<10}"
            )

    # -------------------------- parsing helpers --------------------------
    def clear_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self):
        for p in [self.config_path, self.cache_dir, self.work_dir]:
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink(missing_ok=True)
            except Exception:
                # Best-effort cleanup; tuning result should not be affected by cleanup failures.
                pass

    def round_to_multiple_of_4(self, value: int) -> int:
        return ((value + 3) // 4) * 4

    def parse_output(self, output: str) -> Tuple[Tuple[int, ...], Tuple[int, ...], bool]:
        branch_usage: Tuple[int, ...] = ()
        branch_available: Tuple[int, ...] = ()
        has_vgpr_usage_before_branch = False

        num_vgpr_pattern = r"BranchNumVGPRs\[(\d+)\]\s*=\s*(\d+)"
        for match in re.finditer(num_vgpr_pattern, output):
            branch_id = int(match.group(1))
            vgpr_usage = int(match.group(2))
            branch_usage += (vgpr_usage,)

        available_vgpr_pattern = r"BranchAvailableVGPRs\[(\d+)\]\s*=\s*(\d+)"
        for match in re.finditer(available_vgpr_pattern, output):
            branch_id = int(match.group(1))
            available = int(match.group(2))
            branch_available += (available,)

        if re.search(r"Warning:\s*found\s*vgpr\s*before\s*wave\s*branch", output, re.IGNORECASE):
            has_vgpr_usage_before_branch = True

        return branch_usage, branch_available, has_vgpr_usage_before_branch

    def check_assembly_spill(self) -> Tuple[bool, int]:
        spill_count = 0
        found_assembly = False

        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for asm_file in subdir.rglob("*.amdgcn"):
                    found_assembly = True
                    try:
                        with open(asm_file, "r") as f:
                            content = f.read()
                        pattern = r"\.vgpr_spill_count:\s*(\d+)"
                        match = re.search(pattern, content)
                        if match:
                            count = int(match.group(1))
                            spill_count = max(spill_count, count)
                            self._log_info(f"Spill count: {spill_count}")
                    except Exception as e:
                        self._log_warning(f"Failed to read assembly file {asm_file}: {e}")

        return found_assembly, spill_count

    # -------------------------- allocation mapping --------------------------
    def branch_allocation_to_partition_allocation(
        self, mode: str, branch_allocation: Tuple[int, ...]
    ) -> Dict[str, int]:
        partition_allocation: Dict[str, int] = {}
        for branch, partition in self.branch_to_partition[mode].items():
            partition_allocation[partition] = max(
                partition_allocation.get(partition, 0), branch_allocation[branch]
            )
        return partition_allocation

    def update_config(self, branch_allocation: Tuple[int, ...], mode: str):
        if self.config.get("wasp_enabled", False) and self.config.get("wdra_enabled", False):
            partition_allocation = self.branch_allocation_to_partition_allocation(
                mode, branch_allocation
            )
            if "load" in partition_allocation:
                self.config["wdra_num_load_regs"] = partition_allocation["load"]
            if "mma_main" in partition_allocation:
                self.config["wdra_num_mma_regs_main"] = partition_allocation["mma_main"]
            # 始终传 tail：单 mma 分支时固定为 0（不作为一路 wave 分支参与 validate_allocation）
            if self.num_mma_branches == 1:
                self.config["wdra_num_mma_regs_tail"] = 0
            elif "mma_tail" in partition_allocation:
                self.config["wdra_num_mma_regs_tail"] = partition_allocation["mma_tail"]

        with open(self.config_path, "w") as f:
            json.dump(self.config, f)

    def run_script(self) -> Tuple[str, str, int]:
        self.clear_cache()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TRITON_CACHE_DIR"] = str(self.cache_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                ["python", "-u", self.script_path, "--json-config", self.config_path, "--compile-only"],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(self.work_dir),
                timeout=300,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Script execution timeout", -1
        except Exception as e:
            return "", str(e), -1

    def test_allocation(self) -> Tuple[bool, bool, int, str, int]:
        stdout, stderr, returncode = self.run_script()
        output = stdout + stderr

        if self.verbose:
            self._log_section("Script Output", level=2)
            print(output)

        branch_usage, branch_available, has_vgpr_usage_before_branch = self.parse_output(output)
        self._last_script_output = output

        _, spill_count = self.check_assembly_spill()

        has_overflow = spill_count > 0
        has_warning = has_vgpr_usage_before_branch

        if returncode != 0:
            self._log_error(f"Script failed (returncode={returncode})", level=1)
        elif has_overflow:
            self._log_warning(f"Spill detected: {spill_count} spills", level=1)
        elif has_warning:
            self._log_warning("VGPR usage before branch detected", level=1)

        return has_overflow, has_warning, spill_count, output, returncode

    # -------------------------- search space --------------------------
    def get_initial_allocation(self) -> Tuple[int, Tuple[int, ...]]:
        min_requirements = [self.round_to_multiple_of_4(usage) for usage in self.initial_branch_usage]

        for idx, alloc_tuple in enumerate(self.all_valid_allocations):
            if not all(alloc >= min_req for alloc, min_req in zip(alloc_tuple, min_requirements)):
                continue
            return idx, alloc_tuple

        return 0, self.all_valid_allocations[0]

    def validate_allocation(self, branch_allocation: Tuple[int, ...]) -> bool:
        """
        与后端约束一致：每个「硬件 wave 分支」分配值为 4 的倍数且 <=256。
        两路 mma（三路 wave）时 main/tail 对应分支可不同。
        各路之和须能被 num_branches 整除，且平均 VGPR 为 4 的倍数。

        注意：单 mma 时 JSON 仍会传 wdra_num_mma_regs_tail=0，但该占位不增加 wave 分支数，
        此处 branch_allocation 长度仍为 num_branches（仅 load + main），不把 tail 当作一路分支校验。
        """
        for vgpr in branch_allocation:
            if vgpr % 4 != 0 or vgpr > 256:
                return False

        total = sum(branch_allocation)
        if total % self.num_branches != 0:
            return False

        avg = total / self.num_branches
        if avg % 4 != 0:
            return False

        return True

    # -------------------------- mapping establishment --------------------------
    def establish_mapping(self) -> Tuple[bool, Dict[int, str]]:
        self._log_stage("Step 1: Establishing Branch-to-Partition Mapping")
        # 用互不相同的 BranchAvailableVGPRs 区分 load / main mma / tail mma。
        # 单 mma：probe 仍传 tail=0；匹配只用 main=16 与 load=64（勿把 tail=0 加入 test_values，避免与 0 误匹配）。
        # 双 mma：main=16, tail=28, load=64（sum=108, 3 分支 avg=36）。
        if self.num_mma_branches == 1:
            test_values = OrderedDict([("mma_main", 16), ("load", 64)])
        else:
            test_values = OrderedDict([("mma_main", 16), ("mma_tail", 28), ("load", 64)])
        self._log_info(f"Test values (WDRA, local-wave): {test_values}")

        mode = "local-wave"

        probe_config = dict(self.config)
        probe_config["wdra_num_load_regs"] = test_values["load"]
        probe_config["wdra_num_mma_regs_main"] = test_values["mma_main"]
        probe_config["wdra_num_mma_regs_tail"] = (
            0 if self.num_mma_branches == 1 else test_values["mma_tail"]
        )

        with open(self.config_path, "w") as f:
            json.dump(probe_config, f)

        stdout, stderr, returncode = self.run_script()
        output = stdout + stderr
        self._last_script_output = output
        if self.verbose:
            self._log_info("Full output:", level=1)
            print(output)

        branch_usage, branch_available, _ = self.parse_output(output)
        self.initial_branch_usage = branch_usage
        if len(branch_usage) != self.num_branches or len(branch_available) != self.num_branches:
            self._log_error("Failed to parse VGPR usage information from output")
            if returncode != 0:
                self._log_error("Script failed and no VGPR information found. Cannot continue.")
            self._fail(
                f"parse branch VGPR failed (usage={len(branch_usage)}, available={len(branch_available)}, "
                f"need {self.num_branches} each; rc={returncode})",
                with_output=True,
            )
            return False, {}

        available_vgpr_pattern = r"BranchAvailableVGPRs\[(\d+)\]\s*=\s*(\d+)"

        output_branches: Dict[int, int] = {}
        for match in re.finditer(available_vgpr_pattern, output):
            output_branch_id = int(match.group(1))
            available = int(match.group(2))
            output_branches[output_branch_id] = available

        mapping: Dict[int, str] = {}
        for output_branch_id, available_value in output_branches.items():
            for partition, value in test_values.items():
                if available_value == value:
                    mapping[output_branch_id] = partition
                    break

        self.branch_to_partition[mode] = mapping
        part_lists: Dict[str, List[int]] = {"load": [], "mma_main": [], "mma_tail": []}
        for branch_id, partition in mapping.items():
            if partition in part_lists:
                part_lists[partition].append(branch_id)
        self.partition_to_branch[mode] = part_lists

        self._print_branch_summary(
            branch_usage=branch_usage, branch_available=branch_available, mode=mode, level=1
        )

        self._log_success("Mapping established successfully")
        return True, mapping

    # -------------------------- search refinement --------------------------
    def try_reduce_allocation(
        self,
        alloc_idx: int,
        branch_usage: Tuple[int, ...],
        forbidden: Optional[Set[int]] = None,
    ) -> Tuple[bool, int, Tuple[int, ...]]:
        self._log_info("Optimizing: reducing allocation...")
        alloc_tuple = self.all_valid_allocations[alloc_idx]
        skip = forbidden or set()

        for idx in range(0, alloc_idx):
            if idx in skip:
                continue
            alloc = self.all_valid_allocations[idx]
            meet_requirements = all(alloc_val >= usage for alloc_val, usage in zip(alloc, branch_usage))
            if meet_requirements:
                return True, idx, alloc

        self._log_info("Cannot reduce allocation further (or all smaller candidates previously spilled)")
        return False, alloc_idx, alloc_tuple

    def try_increase_allocation(
        self,
        alloc_idx: int,
        spill_count: int,
        branch_available: Tuple[int, ...],
        branch_usage: Tuple[int, ...],
    ) -> Tuple[bool, int, Tuple[int, ...] | None]:
        """
        按汇编 spill_count 估计需要增加的 VGPR（overflow 分支上增量 >= spill_count）。
        若找不到满足该启发式的更大分配（含最大分配仍不满足），调用方应改用最大分配再试，
        因为 spill_count 不一定可靠。
        """
        self._log_info("Trying to increase allocation (spill_count heuristic)...")

        current_alloc = self.all_valid_allocations[alloc_idx]
        overflow_branches = [
            i for i in range(len(branch_available)) if branch_available[i] == branch_usage[i]
        ]

        for idx, alloc in enumerate(self.all_valid_allocations):
            if idx <= alloc_idx:
                continue
            all_meet_requirements = all(
                alloc_val >= min_requirement
                for alloc_val, min_requirement in zip(alloc, branch_usage)
            )
            overflow_increase = sum(alloc[i] - current_alloc[i] for i in overflow_branches)
            if all_meet_requirements and overflow_increase >= spill_count:
                self._log_info(f"Found larger allocation: idx={idx}, {alloc}")
                return True, idx, alloc

        self._log_warning(
            "No allocation satisfies spill_count heuristic (including larger indices); "
            "caller may try max allocation — spill_count can be unreliable."
        )
        return False, alloc_idx, None

    # -------------------------- main WDRA loop --------------------------
    def run_wdra(self, alloc_idx: int, *, max_spill_fallback_used: bool = False):
        self._log_stage("Step 2: Testing Local-Wave Mode")
        alloc_tuple = self.all_valid_allocations[alloc_idx]
        self._log_info(f"Allocation: idx={alloc_idx}, {alloc_tuple}")
        self.update_config(alloc_tuple, "local-wave")
        has_overflow, has_warning, spill_count, output, returncode = self.test_allocation()
        branch_usage, branch_available, _ = self.parse_output(output)
        self._print_branch_summary(
            branch_usage=branch_usage, branch_available=branch_available, mode="local-wave", level=1
        )

        if not has_overflow and not has_warning and returncode == 0:
            self._log_success("Local-wave mode works!")
            success, best_alloc_idx, best_alloc_tuple = self.try_reduce_allocation(
                alloc_idx, branch_usage, forbidden=self._alloc_indices_spilled
            )
            if not success:
                self._log_info("Cannot reduce allocation further. Consider done.")
                self._log_success(f"Final: idx={best_alloc_idx}, {best_alloc_tuple}")
                self.best_alloc_tuple = best_alloc_tuple
                return
            else:
                self._log_info("Retrying with reduced allocation in local-wave mode")
                return self.run_wdra(best_alloc_idx, max_spill_fallback_used=max_spill_fallback_used)

        elif has_overflow:
            self._alloc_indices_spilled.add(alloc_idx)
            self._log_warning(f"Overflow (spills={spill_count}) → try increasing VGPRs (heuristic)")
            success, best_alloc_idx, best_alloc_tuple = self.try_increase_allocation(
                alloc_idx, spill_count, branch_available, branch_usage
            )
            if success:
                self._log_info("Retrying with increased allocation in local-wave mode")
                return self.run_wdra(best_alloc_idx, max_spill_fallback_used=max_spill_fallback_used)

            # 启发式找不到「增量 >= spill_count」的更大分配时，仍用最大分配编译一次再按实测加减
            max_idx = len(self.all_valid_allocations) - 1
            if alloc_idx < max_idx and not max_spill_fallback_used:
                self._log_warning(
                    "Spill heuristic did not match any larger allocation; "
                    f"compiling once with maximum allocation (idx={max_idx}) — "
                    "spill_count may not reflect actual register pressure."
                )
                return self.run_wdra(max_idx, max_spill_fallback_used=True)

            # 已在最大分配或已尝试过 max 回退：按实测再减（不再依赖 spill_count）
            self._log_warning(
                "Overflow persists; trying to reduce allocation while meeting usage (measured spill may be misleading)."
            )
            success, best_alloc_idx, best_alloc_tuple = self.try_reduce_allocation(
                alloc_idx, branch_usage, forbidden=self._alloc_indices_spilled
            )
            if success:
                self._log_info("Retrying with reduced allocation after spill")
                return self.run_wdra(best_alloc_idx, max_spill_fallback_used=max_spill_fallback_used)

            self._log_error("VGPR spill persists; cannot increase (heuristic/max) or reduce further.")
            self._fail(
                f"VGPR spill (count={spill_count}) at allocation idx={alloc_idx}",
                with_output=True,
            )
            return

        elif has_warning:
            self._log_error("VGPR usage before branch detected. Cannot continue.")
            self._fail("VGPR usage before wave branch", with_output=True)
            return

        else:
            self._log_error("Runtime error. Cannot continue.")
            self._fail(f"script/runtime error (rc={returncode})", with_output=True)
            return

    # -------------------------- public API --------------------------
    def tune(self) -> Tuple[bool, Dict[str, Any], str]:
        self.last_error = ""
        self._log_header("VGPR Tuning Process")

        mapping_ok, _ = self.establish_mapping()
        if not mapping_ok:
            self._log_error("Failed to establish branch-to-partition mapping. Aborting.")
            if not self.last_error:
                self._fail("establish_mapping failed", with_output=True)
            return False, self.config, self.last_error

        self._log_stage("Step 2: Finding Initial VGPR Allocation")
        initial_idx, initial_allocation = self.get_initial_allocation()
        self._log_success(f"Initial allocation: idx={initial_idx}, {initial_allocation}")

        self._alloc_indices_spilled.clear()
        self.run_wdra(initial_idx)

        self._log_header("Tuning Result")
        if self.best_alloc_tuple is not None:
            self._log_success("VGPR tuning completed successfully!")
            partition_alloc = self.branch_allocation_to_partition_allocation(
                "local-wave", self.best_alloc_tuple
            )
            self._log_info(f"Final branch allocation: {self.best_alloc_tuple}")
            self._log_info(f"Final partition allocation: {partition_alloc}")
            return True, self.config, ""

        self._log_error("VGPR tuning failed. Could not find a working allocation.")
        if not self.last_error:
            self._fail("no valid VGPR allocation found after search", with_output=True)
        return False, self.config, self.last_error


def _archive_triton_cache_on_failure(
    cache_dir: Path, logs_dir: str, config_idx: int, log_path: str
) -> None:
    """失败时将本 config 使用的 TRITON_CACHE_DIR 复制到 tune_wdra_logs，便于排查（在 cleanup 删除前调用）。"""
    dest = Path(logs_dir) / f"config_{config_idx}_triton_cache"
    try:
        if not cache_dir.exists():
            return
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cache_dir, dest)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n[triton cache] snapshot saved to: {dest}\n")
    except Exception as ex:
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n[triton cache] archive failed: {ex}\n")
        except Exception:
            pass


def _tune_one_config(
    config_idx: int,
    n: int,
    config: Dict[str, Any],
    script_path: str,
    logs_dir: str,
    verbose: bool,
) -> Dict[str, Any]:
    log_path = os.path.join(logs_dir, f"config_{config_idx}.log")
    result: Dict[str, Any] = {
        "config_idx": config_idx,
        "ok": False,
        "skip": False,
        "final_config": config,
        "err": "",
        "status_line": "",
    }

    if not (config.get("wasp_enabled", False) and config.get("wdra_enabled", False)):
        result["skip"] = True
        result["ok"] = True
        result["status_line"] = f"config {config_idx}/{n}: skipped"
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"config {config_idx}/{n}: skipped\n")
            lf.write("reason: WASP or WDRA not enabled\n")
            lf.write(f"config: {json.dumps(config, ensure_ascii=False, indent=2)}\n")
        return result

    wasp_load = config.get("wasp_num_load_warps")
    wasp_mma = config.get("wasp_num_mma_warps")
    if wasp_load != 4 or wasp_mma not in (4, 8):
        result["skip"] = True
        result["ok"] = True
        result["status_line"] = f"config {config_idx}/{n}: skipped"
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"config {config_idx}/{n}: skipped\n")
            lf.write(
                "reason: unsupported WASP wave config "
                f"(wasp_num_load_warps={wasp_load}, wasp_num_mma_warps={wasp_mma})\n"
            )
            lf.write(f"config: {json.dumps(config, ensure_ascii=False, indent=2)}\n")
        return result

    wdra_tuner = WDRATuner(script_path, config=config, verbose=verbose)
    wdra_tuner.verbose = True
    ok = False
    final_config = config
    err = ""
    try:
        with open(log_path, "w", encoding="utf-8") as lf, redirect_stdout(lf), redirect_stderr(lf):
            print(f"config {config_idx}/{n} raw log")
            print(f"input config: {json.dumps(config, ensure_ascii=False)}")
            ok, final_config, err = wdra_tuner.tune()
            print(f"result: ok={ok}")
            if err:
                print(f"error: {err}")
            print(f"final config: {json.dumps(final_config, ensure_ascii=False)}")
    finally:
        if not ok:
            _archive_triton_cache_on_failure(wdra_tuner.cache_dir, logs_dir, config_idx, log_path)
        wdra_tuner.cleanup()

    result["ok"] = ok
    result["final_config"] = final_config if ok else config
    result["err"] = err
    if ok:
        cfg_line = json.dumps(result["final_config"], ensure_ascii=False, separators=(",", ":"))
        result["status_line"] = f"config {config_idx}/{n}: success {cfg_line}"
    else:
        result["status_line"] = f"config {config_idx}/{n}: failed"
    return result


def main(configs):
    if len(sys.argv) < 2:
        print("Usage: python tune_wdra.py <target_script.py> [--verbose|-v]")
        print("  Default: concise (only config index + success/fail per config)")
        print("  -v / --verbose: full tuner logs")
        sys.exit(1)

    verbose = False
    jobs = min((os.cpu_count() or 1), 8)
    for arg in sys.argv[2:]:
        if arg in ["--verbose", "-v"]:
            verbose = True
        elif arg.startswith("--jobs="):
            jobs = max(1, int(arg.split("=", 1)[1]))
        elif arg == "--jobs":
            # Support: --jobs N
            # Parsed in a second pass for simplicity.
            pass
    # Parse "--jobs N" form.
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == "--jobs" and i + 1 < len(sys.argv):
            jobs = max(1, int(sys.argv[i + 1]))

    script_path = sys.argv[1]
    if not os.path.exists(script_path):
        print(f"error: script not found: {script_path}")
        sys.exit(1)

    n = len(configs)
    logs_dir = os.path.join(os.path.dirname(__file__), "tune_wdra_logs")
    os.makedirs(logs_dir, exist_ok=True)
    if verbose:
        print("=" * 70)
        print("VGPR Size Auto-Tuning Tool (WASP+WDRA)")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total configurations to tune: {n}")
        print("=" * 70)

    final_configs: List[Dict[str, Any]] = []

    # Verbose mode keeps the original serial behavior for readable console output.
    if verbose or jobs == 1:
        # 显式 update + close，避免 tqdm(enumerate(...)) 在循环结束后不刷新/不收尾导致进度条卡住
        pbar = tqdm(total=n, desc="Tuning configs", unit="config", disable=not verbose)
        try:
            for config_idx, config in enumerate(configs, start=1):
                if verbose:
                    print("\n" + "=" * 70)
                    print(f"Processing Configuration {config_idx}/{n}")
                    print("=" * 70)

                single = _tune_one_config(config_idx, n, config, script_path, logs_dir, verbose)
                if single["ok"]:
                    final_configs.append(single["final_config"])
                    if not verbose:
                        print(single["status_line"])
                    else:
                        print(f"\n[✓ SUCCESS] Configuration {config_idx} tuning completed successfully")
                else:
                    if verbose:
                        print(f"\n[✗ ERROR] Configuration {config_idx} tuning failed")
                        if single["err"]:
                            print(f"  reason: {single['err']}")
                pbar.update(1)
        finally:
            pbar.close()
        # Serial path done.
    else:
        # Concise mode: run all configs in parallel.
        # Linux 默认 fork 子进程 + ROCm/编译子进程时，worker 常无法正常退出，ProcessPoolExecutor
        # 在 __exit__ 里 shutdown(wait=True) 会长时间卡住；tqdm 已 100% 但终端像冻住。
        # 使用 spawn 避免 fork 继承父进程状态，并在 shutdown 前先收尾 tqdm + flush stderr。
        results_by_idx: Dict[int, Dict[str, Any]] = {}
        mp_ctx = multiprocessing.get_context("spawn")
        ex = ProcessPoolExecutor(max_workers=jobs, mp_context=mp_ctx)
        try:
            future_list = [
                ex.submit(_tune_one_config, idx, n, cfg, script_path, logs_dir, False)
                for idx, cfg in enumerate(configs, start=1)
            ]
            pbar = tqdm(
                total=n,
                desc="Tuning configs",
                unit="config",
                miniters=1,
                file=sys.stderr,
                dynamic_ncols=True,
            )
            try:
                for fut in as_completed(future_list):
                    res = fut.result()
                    results_by_idx[res["config_idx"]] = res
                    pbar.update(1)
            finally:
                pbar.refresh()
                pbar.close()
                sys.stderr.flush()
        finally:
            ex.shutdown(wait=True)

        for idx in range(1, n + 1):
            res = results_by_idx[idx]
            if res["ok"]:
                final_configs.append(res["final_config"])
            print(res["status_line"])

    # old serial implementation retained below was replaced by serial+parallel paths

    output_file = os.path.join(os.path.dirname(__file__), "final_configs.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_configs, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n" + "=" * 70)
        print("Tuning Summary")
        print("=" * 70)
        print(f"[INFO] Total configurations processed: {n}")
        print(f"[INFO] Successful configurations: {len(final_configs)}")
        print(f"[INFO] Failed configurations: {n - len(final_configs)}")
        print(f"[INFO] Results saved to: {output_file}")
        print(f"[INFO] Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    else:
        print(f"saved {len(final_configs)} configs -> {output_file}")
        print(f"logs saved to: {logs_dir}")


if __name__ == "__main__":
    configs = [
        {
            "BLOCK_SIZE_M": BM,
            "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK,
            "GROUP_SIZE_M": GM,
            "wasp_enabled": True,
            "wdra_enabled": True,
            "wasp_num_load_warps": 4,
            "wasp_num_mma_warps": num_mma_warps,
        }
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
        for BK in [32, 64, 128]
        for GM in [1]
        for num_mma_warps in [4, 8]
    ]
    main(configs)

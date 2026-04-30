r"""
`hcuify.py` is a script that convert AMDGPU backend to HCU backend.

It replaces "AMD", "AMDGPU", "amd", and "amdgpu" in the file with
"HCU", "HCUGPU", "hcu", and "hcugpu". It also replaces the keyword
AMD with HCU in directory and file names.

Note: not all AMD-related keywords are replaced. AMD keywords used in
LLVM code are preserved, since LLVM still uses the amdgcn backend.
User-specified keywords are also preserved, such as AMDMfmaEncodingAttr
(HCUMfmaEncodingAttr is not defined yet).

**How to use this script:**

::

    >>> python hcuify.py third_party/amd third_party/hcu
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


TEXT_EXTS = {
    ".h", ".hpp", ".hh", ".hxx", ".inc",
    ".c", ".cc", ".cpp", ".cxx",
    ".cu", ".cuh",
    ".td", ".mlir", ".py", ".txt", ".cmake",
    ".md", ".rst", ".yaml", ".yml", ".json",
}

SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "build", ".cache", "__pycache__",
}

NAME_PATTERNS = [
    (re.compile(r"AMDGPU"), "HCUGPU"),
    (re.compile(r"amdgpu"), "hcugpu"),
    (re.compile(r"AMD"), "HCU"),
    (re.compile(r"amd"), "hcu"),
]

# (re.compile(r"amd(?!gcn|hsa)"), "hcu"),
# CONTENT_TOKEN_RE = re.compile(r"AMDGPU|amdgpu|AMD|amd(?!gcn|hsa)")
CONTENT_TOKEN_RE = re.compile(r"AMDGPU|amdgpu|AMD|amd")
LLVM_CHAIN_RE = re.compile(
    r"\bllvm::[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*"
)

# Since LLVM still uses the AMD backend.
# Do not perform keyword replacement on content matching the rules below.
PROTECTED_CONTENT_RE = re.compile(
    r"\bllvm::[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*"
    r"|"
    r'#include\s*[<"]mlir/Dialect/AMD[^>"]+[>"]'
    r"|"
    r"\bmlir::gpu::amd"
    r"|"
    r"\bAMDW[Mm][Mm][Aa][A-Za-z0-9_]*\b" 
    r"|"
    r"\bAMDM[Ff][Mm][Aa][A-Za-z0-9_]*\b" 
    r"|"
    r"\b[A-Za-z0-9_]*AMDRotating[A-Za-z0-9_]*\b"
    r"|"
    r"\blibamd.*"
    r"|"
    r"\bCALLING_CONV_AMDGPU_KERNEL\b"
    r"|"
    r"\b__HIP_PLATFORM_AMD__\b"
    r"|"
    r'\badd_fn_attr\(\s*"[^"\\]*(?:\\.[^"\\]*)*"'
    r"|"
    r'\bremove_fn_attr\(\s*"[^"\\]*(?:\\.[^"\\]*)*"'
    r"|"
    r"\b[A-Za-z0-9_\-]*amdgcn[A-Za-z0-9_\-]*\b"
    r"|"
    r"\b[A-Za-z0-9_\-]*amdhsa[A-Za-z0-9_\-]*\b"
    r"|"
    r"\bamdgpu_kernel\b"
)


def is_under_protected_dir(path: Path, root: Path) -> bool:
    PROTECTED_DIR_NAMES = {"hip", "roctracer", "hsa"}
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(part in PROTECTED_DIR_NAMES for part in rel.parts)


@dataclass(frozen=True)
class RenameOp:
    src: Path
    dst: Path
    kind: str  # "dir" or "file"


def is_probably_text_file(path: Path) -> bool:
    return path.suffix in TEXT_EXTS


def replace_name(name: str) -> str:
    out = name
    for pat, repl in NAME_PATTERNS:
        out = pat.sub(repl, out)
    return out


def replace_token(token: str) -> str:
    if token == "AMDGPU":
        return "HCUGPU"
    if token == "amdgpu":
        return "hcugpu"
    if token == "AMD":
        return "HCU"
    if token == "amd":
        return "hcu"
    return token


def should_skip_token_after_mlir(text: str, start: int) -> bool:
    i = start - 1

    while i >= 0 and text[i].isspace():
        i -= 1

    if i < 1 or text[i] != ":" or text[i - 1] != ":":
        return False

    i -= 2
    while i >= 0 and text[i].isspace():
        i -= 1

    end = i
    while i >= 0 and (text[i].isalnum() or text[i] == "_"):
        i -= 1

    prev_ident = text[i + 1:end + 1]
    return prev_ident == "mlir"


def replace_text_segment(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        token = m.group(0)
        if should_skip_token_after_mlir(text, m.start()):
            return token
        return replace_token(token)

    return CONTENT_TOKEN_RE.sub(repl, text)


def replace_all_text(text: str) -> str:
    parts = []
    last = 0

    for m in PROTECTED_CONTENT_RE.finditer(text):
        start, end = m.span()
        if start > last:
            parts.append(replace_text_segment(text[last:start]))
        parts.append(text[start:end])  # 保护区原样保留
        last = end

    if last < len(text):
        parts.append(replace_text_segment(text[last:]))

    return "".join(parts)


def process_file_content(path: Path, dry_run: bool = False) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False
    except Exception:
        return False

    new_content = replace_all_text(content)
    if new_content == content:
        return False

    if not dry_run:
        path.write_text(new_content, encoding="utf-8")
    return True


def iter_text_files(root: Path):
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if is_under_protected_dir(p, root):
            continue
        if p.is_file() and is_probably_text_file(p):
            yield p


def list_dirs_bottom_up(root: Path):
    dirs = []
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_dir():
            dirs.append(p)
    dirs.sort(key=lambda x: len(x.parts), reverse=True)
    return dirs


def list_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            files.append(p)
    return files


def copy_source_tree(src: Path, dst: Path, dry_run: bool = False):
    if dry_run:
        print(f"[hcuify][dry-run] copy tree: {src} -> {dst}")
        return

    if dst.exists():
        raise RuntimeError(f"[hcuify] destination already exists: {dst}")

    shutil.copytree(src, dst)


def build_dir_rename_plan(root: Path) -> list[RenameOp]:
    ops = []
    for p in list_dirs_bottom_up(root):
        if is_under_protected_dir(p, root):
            continue
        new_name = replace_name(p.name)
        if new_name != p.name:
            ops.append(RenameOp(src=p, dst=p.with_name(new_name), kind="dir"))
    return ops


def build_file_rename_plan(root: Path) -> list[RenameOp]:
    ops = []
    for p in list_files(root):
        if is_under_protected_dir(p, root):
            continue
        new_name = replace_name(p.name)
        if new_name != p.name:
            ops.append(RenameOp(src=p, dst=p.with_name(new_name), kind="file"))
    return ops


def detect_rename_conflicts(ops: list[RenameOp]) -> list[str]:
    conflicts = []
    dst_to_srcs: dict[Path, list[Path]] = {}

    src_set = {op.src for op in ops}

    for op in ops:
        dst_to_srcs.setdefault(op.dst, []).append(op.src)

    for dst, srcs in dst_to_srcs.items():
        if len(srcs) > 1:
            conflicts.append(
                f"[hcuify] rename conflict: multiple sources map to same target: "
                f"{', '.join(map(str, srcs))} -> {dst}"
            )

    for op in ops:
        if op.dst.exists() and op.dst not in src_set:
            conflicts.append(
                f"[hcuify] rename conflict: target already exists: {op.src} -> {op.dst}"
            )

    return conflicts


def execute_rename_plan(ops: list[RenameOp], dry_run: bool = False):
    changed = 0
    if not ops:
        return changed

    prefix = "[hcuify][dry-run]" if dry_run else "[hcuify]"

    if dry_run:
        for op in ops:
            print(f"{prefix} {op.kind} renamed: {op.src} -> {op.dst}")
        return len(ops)

    temp_ops: list[tuple[Path, Path, Path, str]] = []
    nonce = uuid.uuid4().hex

    for idx, op in enumerate(ops):
        temp_name = f".hcuify_tmp_{nonce}_{idx}_{op.src.name}"
        temp_path = op.src.with_name(temp_name)
        if temp_path.exists():
            raise RuntimeError(f"[hcuify] temporary path already exists: {temp_path}")
        op.src.rename(temp_path)
        temp_ops.append((op.src, temp_path, op.dst, op.kind))

    for original_src, temp_path, final_dst, kind in temp_ops:
        if final_dst.exists():
            raise RuntimeError(
                f"[hcuify] rename conflict during finalization: {original_src} -> {final_dst}"
            )
        temp_path.rename(final_dst)
        print(f"{prefix} {kind} renamed: {original_src} -> {final_dst}")
        changed += 1

    return changed


def run_rename_stage(
    ops: list[RenameOp],
    dry_run: bool = False,
):
    if not ops:
        return 0, 0

    conflicts = detect_rename_conflicts(ops)

    if conflicts:
        conflict_srcs = set()
        conflict_dsts = set()

        dst_count: dict[Path, int] = {}
        src_set = {op.src for op in ops}
        for op in ops:
            dst_count[op.dst] = dst_count.get(op.dst, 0) + 1

        for op in ops:
            if dst_count[op.dst] > 1:
                conflict_srcs.add(op.src)
                conflict_dsts.add(op.dst)
            elif op.dst.exists() and op.dst not in src_set:
                conflict_srcs.add(op.src)
                conflict_dsts.add(op.dst)

        for msg in conflicts:
            print(msg)

        filtered_ops = [op for op in ops if op.src not in conflict_srcs and op.dst not in conflict_dsts]
        changed = execute_rename_plan(filtered_ops, dry_run=dry_run)
        return changed, len(conflicts)

    changed = execute_rename_plan(ops, dry_run=dry_run)
    return changed, 0


def process_target_tree(root: Path, dry_run: bool = False):
    scanned_files = 0
    changed_files = 0

    for path in iter_text_files(root):
        scanned_files += 1
        changed = process_file_content(path, dry_run=dry_run)
        if changed:
            changed_files += 1
            rel = path.relative_to(root)
            prefix = "[hcuify][dry-run]" if dry_run else "[hcuify]"
            print(f"{prefix} content changed: {rel}")

    dir_ops = build_dir_rename_plan(root)
    renamed_dirs, dir_conflicts = run_rename_stage(
        dir_ops,
        dry_run=dry_run,
    )

    file_ops = build_file_rename_plan(root)
    renamed_files, file_conflicts = run_rename_stage(
        file_ops,
        dry_run=dry_run,
    )

    print(f"[hcuify] scanned text files: {scanned_files}")
    print(f"[hcuify] content changed files: {changed_files}")
    print(f"[hcuify] renamed directories: {renamed_dirs}")
    print(f"[hcuify] renamed files: {renamed_files}")
    print(f"[hcuify] rename conflicts: {dir_conflicts + file_conflicts}")


def build_dir_rename_plan_dry_run(src_root: Path, dst_root: Path) -> list[RenameOp]:
    ops = []
    for p in list_dirs_bottom_up(src_root):
        rel = p.relative_to(src_root)
        new_rel = Path(*(replace_name(part) for part in rel.parts))
        if new_rel != rel:
            ops.append(RenameOp(src=dst_root / rel, dst=dst_root / new_rel, kind="dir"))
    return ops


def build_file_rename_plan_dry_run(src_root: Path, dst_root: Path) -> list[RenameOp]:
    ops = []
    for p in list_files(src_root):
        rel = p.relative_to(src_root)
        renamed_parent = Path(*(replace_name(part) for part in rel.parent.parts))
        old_in_dst = dst_root / renamed_parent / rel.name if rel.parent.parts else dst_root / rel.name
        new_name = replace_name(rel.name)
        if new_name != rel.name:
            new_in_dst = old_in_dst.with_name(new_name)
            ops.append(RenameOp(src=old_in_dst, dst=new_in_dst, kind="file"))
    return ops


def process_target_tree_dry_run(src_root: Path, dst_root: Path):
    scanned_files = 0
    changed_files = 0

    print(f"[hcuify][dry-run] copy tree: {src_root} -> {dst_root}")

    for path in iter_text_files(src_root):
        scanned_files += 1
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
        except Exception:
            continue

        new_content = replace_all_text(content)
        if new_content != content:
            rel = path.relative_to(src_root)
            new_rel = Path(*(replace_name(part) for part in rel.parts))
            changed_files += 1
            print(f"[hcuify][dry-run] content changed: {dst_root / new_rel}")

    dir_ops = build_dir_rename_plan_dry_run(src_root, dst_root)
    renamed_dirs, dir_conflicts = run_rename_stage(
        dir_ops,
        dry_run=True,
    )

    file_ops = build_file_rename_plan_dry_run(src_root, dst_root)
    renamed_files, file_conflicts = run_rename_stage(
        file_ops,
        dry_run=True,
    )

    print(f"[hcuify] scanned text files: {scanned_files}")
    print(f"[hcuify] content changed files: {changed_files}")
    print(f"[hcuify] renamed directories: {renamed_dirs}")
    print(f"[hcuify] renamed files: {renamed_files}")
    print(f"[hcuify] rename conflicts: {dir_conflicts + file_conflicts}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy source tree to destination, then replace file contents and rename "
            "directories/files recursively: AMDGPU->HCUGPU, amdgpu->hcugpu, "
            "AMD->HCU, amd->hcu; keep amdgcn unchanged. "
            "In file contents, llvm::... chains and selected ttg encoding attrs "
            "are preserved; for mlir::..., only the token immediately following "
            "mlir:: is preserved."
        )
    )
    parser.add_argument("src_dir", help="Source directory")
    parser.add_argument("dst_dir", help="Destination directory (must not already exist)")
    parser.add_argument("--dry-run", action="store_true", help="Only preview planned changes")
    args = parser.parse_args()

    src = Path(args.src_dir).resolve()
    dst = Path(args.dst_dir).resolve()

    if not src.exists():
        print(f"[hcuify] source dir does not exist: {src}", file=sys.stderr)
        return 1
    if not src.is_dir():
        print(f"[hcuify] source is not a directory: {src}", file=sys.stderr)
        return 1
    if src == dst:
        print("[hcuify] source and destination must be different", file=sys.stderr)
        return 1

    try:
        if args.dry_run:
            process_target_tree_dry_run(
                src_root=src,
                dst_root=dst,
            )
        else:
            copy_source_tree(src, dst, dry_run=False)
            process_target_tree(
                dst,
                dry_run=False,
            )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

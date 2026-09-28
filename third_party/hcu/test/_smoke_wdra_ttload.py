"""Smoke: WDRA 1P2C (4+8) and 2P2C (8+8) for tt.load and MLS."""
import os
import re
import shutil
import tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, warp_specialize=WARP_SPECIALIZE):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run_load(load_w, mma_w, label):
    M = N = K = 128
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    cfg = {
        "wasp_partition_warps": (4,) * ((load_w + mma_w) // 4),
        "wasp_partition_regs": ((80, 96, 144, 160) if load_w == 8 else (88, 144, 140)),
    }
    dump = tempfile.mkdtemp(prefix="wdra_ttload_")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = dump
    try:
        matmul_kernel[(1,)](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            WARP_SPECIALIZE=True, **cfg,
        )
        torch.cuda.synchronize()
        total = starts = partition_warps = None
        for root, _, files in os.walk(dump):
            for f in files:
                if not f.endswith(".ttgir"):
                    continue
                text = open(os.path.join(root, f)).read()
                m = re.search(r'"ttg.total-num-warps" = (\d+)', text)
                if m:
                    total = m.group(1)
                m = re.search(r"warpGroupStartIds = array<i32: ([^>]+)>", text)
                if m:
                    starts = m.group(1)
                m = re.search(r"hcu.wasp_partition_warps = array<i32: ([^>]+)>", text)
                if m:
                    partition_warps = m.group(1)
        print(f"{label}: PASS")
        print(f"  partition-warps={partition_warps} total-num-warps={total} starts={starts}")
        return True
    except Exception as e:
        msg = str(e)
        for line in msg.splitlines():
            if "direct SSA" in line or "error:" in line.lower():
                print(line[:240])
                break
        print(f"{label}: FAIL {type(e).__name__}")
        return False
    finally:
        shutil.rmtree(dump, ignore_errors=True)
        for k in ("TRITON_KERNEL_DUMP", "TRITON_DUMP_DIR"):
            os.environ.pop(k, None)


def main():
    ok = True
    ok &= run_load(4, 8, "tt.load 1P2C 4+8")
    ok &= run_load(8, 8, "tt.load 2P2C 8+8")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

# Triton-Hygon

**基于 [Triton](https://github.com/triton-lang/triton) Upstream Commit [`85400f80bf`](https://github.com/triton-lang/triton/commit/85400f80bf859a34ad7a746ffda877faf80312ab)（`release/3.6.x` 分支）的 Hygon HCU 适配版。**

[Triton](https://github.com/triton-lang/triton) 是用于编写高性能 GPU Kernel 的语言与编译器，官方文档见 [triton-lang.org](https://triton-lang.org)。本仓库在 Upstream Commit `85400f80bf` 基线上，为 **Hygon HCU** 扩展编译流水线、运行时与工具链集成，使 Triton DSL 编写的 Kernel 可在 Hygon DTK 环境下运行。

## 上游与基线

| 项目 | 说明 |
|------|------|
| **Upstream 项目** | [triton-lang/triton](https://github.com/triton-lang/triton) |
| **Upstream 基线 Commit** | [`85400f80bf`](https://github.com/triton-lang/triton/commit/85400f80bf859a34ad7a746ffda877faf80312ab) — `[BACKEND] run remove backward prop until a fix point (#8776)` |
| **基线分支** | `release/3.6.x` |
| **本仓库分支** | `release/3.6.x`（在基线 Commit 之上叠加 Hygon 适配提交） |
| **上游许可证** | [MIT](./LICENSE) |

## 源码构建

需已 `source` **Hygon DTK** 环境。默认**无需自行编译 LLVM**：`setup.py` 会按 `cmake/llvm-hash.txt` 下载预编译包到 `~/.triton/llvm/`。

请使用 **`rocky-x64`** 预编译 LLVM。Ubuntu 等 glibc 较新的环境请显式指定（只改下载包名，不改本机编译目标）：

```bash
export TRITON_LLVM_SYSTEM_SUFFIX=rocky-x64
```

Ubuntu 构建机还需安装 zstd 开发包：

```bash
# Debian / Ubuntu
sudo apt-get install -y libzstd-dev

# RHEL / Rocky（通常已具备；若 CMake 报缺 zstd 再装）
# sudo yum install -y libzstd-devel
```

```bash
cd triton
pip install -r python/requirements.txt
pip install -e .
```

构建 Wheel：

```bash
pip install -r python/requirements.txt
python setup.py bdist_wheel
```

若已有匹配 Hash 的本地 LLVM，或处于离线环境：

```bash
export LLVM_SYSPATH=/path/to/llvm
pip install -e .
```

## 快速示例

```python
import torch
import triton
import triton.language as tl

DEVICE = "cuda"  # DTK PyTorch 下仍为 "cuda"（HIP）


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(f"max diff: {torch.max(torch.abs(output_torch - output_triton))}")
```

## 示例与测试

- [python/tutorials](./python/tutorials) — Upstream Triton 教程
- [third_party/hcu/test](./third_party/hcu/test) — HCU 专项测试与 Kernel 示例
- [python/tutorials/dist](./python/tutorials/dist) — 节点内分布式 GEMM / 通信 Overlap 示例

## 第三方源码

本仓库除 Triton 上游外，还包含 ByteDance Triton-distributed / SIMT、AMD HIP extras，以及部分 tutorial 衍生代码。项目、仓库、版本、Copyright、许可证与本地路径见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

## 致谢

感谢 [Triton](https://github.com/triton-lang/triton) 社区。**Triton-Hygon** 基于 Upstream Commit `85400f80bf` 衍生，并针对 Hygon HCU 做了适配。

## License

本仓库沿用上游 **MIT** 许可证，完整条款见根目录 [LICENSE](./LICENSE)。第三方源码的版权与许可证见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

Modified by Hygon Information Technology Co., Ltd.

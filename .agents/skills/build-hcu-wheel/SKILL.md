---
name: build-hcu-wheel
description: 在 HCU 环境构建 Triton-HCU wheel。当用户要求编译/打包 Triton wheel、生成 manylinux 包，或排查构建环境（DTK/torch/AICC）时使用。
---

# Build HCU Wheel

使用 `scripts/hcu/build_hcu_wheel.sh` 从源码构建 Triton-HCU wheel。脚本假定当前环境已具备 python + DTK + torch + AICC，并在构建前逐项检查，缺失时直接报错。

## 前置环境

1. DTK 已安装并加载：

   ```bash
   source /opt/dtk/env.sh
   ```

   DTK 不在 `/opt/dtk` 时设置 `ROCM_PATH`。

2. Python 构建依赖：脚本检测到 pybind11/Cython/cmake/ninja 缺失时会自动执行
   `pip install -r python/requirements.txt`，无需手动安装。

3. torch 已安装：`python3 -c "import torch; print(torch.__version__)"`。

4. AICC（aillvm）已安装：默认位于 `$ROCM_PATH/aillvm`，可用 `hcu_llvm_installer.sh` 自动检测环境安装。

5. 默认执行 manylinux 修复；`auditwheel` 缺失时脚本会自动 `pip install`，无需手动安装（也可用 `--no-manylinux` 跳过）。

## 用法

```bash
bash scripts/hcu/build_hcu_wheel.sh [options]
```

| 选项 | 说明 |
|---|---|
| `--llvm-repo <url>` | 传入时：文件服务器无预编译包则从该仓库本地编译；不传时由 setup.py 直接下载预编译 LLVM |
| `--file-server-ip <ip>` | 上传本地编译 LLVM 的文件服务器地址 |
| `--file-server-user <user>` | 上传本地编译 LLVM 的文件服务器用户；交互提示时可直接回车沿用 |
| `--file-server-pwd <pwd>` | 文件服务器密码；不传时优先尝试 SSH 免密上传，否则在交互终端提示输入用户名/密码，留空则不上传 |
| `--no-manylinux` | 跳过 auditwheel manylinux 修复 |
| `-h, --help` | 显示帮助 |

本地编译 LLVM 后才会触发上传：已配置到文件服务器的 SSH 公钥时自动免密上传；否则交互终端提示输入用户名和密码，直接回车跳过上传（本地构建的 LLVM 照常使用）。

## 示例

本地构建（文件服务器有预编译 LLVM 时由 setup.py 下载，没有则用默认内网仓库本地编译；离线或已有 LLVM 时用 `LLVM_SYSPATH` 指定）：

```bash
source /opt/dtk/env.sh
bash scripts/hcu/build_hcu_wheel.sh
```

内网/CI 构建（确保 LLVM 预编译包存在，默认做 manylinux 修复）：

```bash
bash scripts/hcu/build_hcu_wheel.sh \
  --llvm-repo "http://gitlab-ci-token:${CI_JOB_TOKEN}@${HCU_LLVM_TRITON_GIT_URL}" \
  --file-server-ip "${FILE_SERVER_IP}" \
  --file-server-user "${FILE_SERVER_USER}" \
  --file-server-pwd "${FILE_SERVER_PWD}"
```

仅本地调试、跳过 manylinux 修复：

```bash
bash scripts/hcu/build_hcu_wheel.sh --no-manylinux
```

## 产物

- 默认输出到 `dist/*.whl`，manylinux 修复后的包名为 `*manylinux*.whl`。
- 脚本结束会打印 wheel 的绝对路径。

## 常见报错

| 报错 | 处理 |
|---|---|
| `torch is not installed` | `pip install torch`（版本需与 DTK 匹配） |
| `DTK not found` / `hipcc is missing` | 安装 DTK 或设置 `ROCM_PATH` |
| `hipcc is not in PATH` | `source /opt/dtk/env.sh` |
| `AICC (aillvm) not found` | `bash hcu_llvm_installer.sh` |
| `failed to install auditwheel` / `patchelf` | 检查 pip 源/网络，或加 `--no-manylinux` 跳过修复 |
| `patchelf 0.12 found, requires >= 0.14.5` | 脚本会自动升级 patchelf；仍报错时确认 PATH 中 `patchelf` 指向 pip 安装的新版本 |
| `pybind11 is still missing` / `Cython is still missing` | 检查 pip 源/网络，或手动 `pip install -r python/requirements.txt` |
| LLVM 下载失败 | 使用内网环境，或 `export LLVM_SYSPATH=/path/to/llvm` / 传 `--llvm-repo` |

## 验证构建结果

```bash
pip install dist/*manylinux*.whl
python -c "import triton; print(triton.__version__)"
# HCU 回归
bash third_party/hcu/test/regression.sh
```

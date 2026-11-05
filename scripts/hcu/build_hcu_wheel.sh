#!/bin/bash

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

CUR_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_HOME="$(cd "${CUR_PATH}/../.." && pwd)"

PYTHON_BIN="python3"
TRITON_DEP_URL="http://42.228.13.241:18000/Jenkins/CompileDep/triton"
LLVM_REPO=""
FILE_SERVER_IP=""
FILE_SERVER_USER=""
FILE_SERVER_PWD=""
RUN_MANYLINUX=1

function usage() {
  cat <<EOF
Usage: bash scripts/hcu/build_hcu_wheel.sh [options]

Build the Triton-HCU wheel. The environment must already provide python,
DTK, torch and AICC (aillvm); this script checks them before building.

Options:
  --llvm-repo <url>          LLVM repo cloned when the prebuilt package is
                             missing on the file server; when omitted, setup.py
                             downloads the prebuilt LLVM package
  --file-server-ip <ip>      File server IP for uploading a locally built LLVM
  --file-server-user <user>  File server user for uploading a locally built LLVM
  --file-server-pwd <pwd>    File server password; when omitted, SSH key auth
                             is tried first, then the password is prompted
                             interactively (empty input skips the upload)
  --no-manylinux             Skip the auditwheel manylinux repair
  -h, --help                 Show this help message

Output: dist/*.whl (manylinux-repaired by default)
EOF
}

function die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llvm-repo)
      [[ $# -ge 2 ]] || die "missing value for --llvm-repo"
      LLVM_REPO="$2"
      shift 2
      ;;
    --file-server-ip)
      [[ $# -ge 2 ]] || die "missing value for --file-server-ip"
      FILE_SERVER_IP="$2"
      shift 2
      ;;
    --file-server-user)
      [[ $# -ge 2 ]] || die "missing value for --file-server-user"
      FILE_SERVER_USER="$2"
      shift 2
      ;;
    --file-server-pwd)
      [[ $# -ge 2 ]] || die "missing value for --file-server-pwd"
      FILE_SERVER_PWD="$2"
      shift 2
      ;;
    --no-manylinux)
      RUN_MANYLINUX=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

function check_dtk() {
  local rocm_path="${ROCM_PATH:-/opt/dtk}"
  [[ -d "${rocm_path}" ]] || die "DTK not found at '${rocm_path}'. Install DTK or set ROCM_PATH."
  [[ -x "${rocm_path}/bin/hipcc" ]] || die "DTK toolchain is incomplete: '${rocm_path}/bin/hipcc' is missing."
  echo "[check] DTK: ${rocm_path}"
  command -v hipcc >/dev/null 2>&1 || echo "[check] WARNING: hipcc is not in PATH; run 'source ${rocm_path}/env.sh' first."
}

function check_aicc() {
  local rocm_path="$1"
  local aillvm_path="${rocm_path}/aillvm"
  [[ -d "${aillvm_path}" ]] || die "AICC (aillvm) not found at '${aillvm_path}'. Install it with: bash hcu_llvm_installer.sh --major <version>"
  local clang=""
  for candidate in clang-22 clang-18 clang; do
    if [[ -x "${aillvm_path}/bin/${candidate}" ]]; then
      clang="${aillvm_path}/bin/${candidate}"
      break
    fi
  done
  [[ -n "${clang}" ]] || die "AICC (aillvm) is incomplete: no clang binary under '${aillvm_path}/bin'."
  echo "[check] AICC: ${aillvm_path} (${clang##*/})"
}

function check_python_env() {
  local torch_version
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python '${PYTHON_BIN}' not found in PATH."
  echo "[check] python: $("${PYTHON_BIN}" -V 2>&1)"

  if ! torch_version="$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)' 2>&1)"; then
    die "torch is not installed for ${PYTHON_BIN}. Install it with: ${PYTHON_BIN} -m pip install torch"
  fi
  echo "[check] torch: ${torch_version}"
}

function install_build_deps() {
  local missing=0
  "${PYTHON_BIN}" -c "import pybind11" >/dev/null 2>&1 || missing=1
  "${PYTHON_BIN}" -c "import Cython" >/dev/null 2>&1 || missing=1
  command -v cmake >/dev/null 2>&1 || missing=1
  command -v ninja >/dev/null 2>&1 || missing=1
  if [[ ${missing} -eq 0 ]]; then
    return
  fi
  echo "[check] installing python build dependencies from python/requirements.txt"
  "${PYTHON_BIN}" -m pip install -r "${SRC_HOME}/python/requirements.txt" || die "failed to install python build dependencies."
  "${PYTHON_BIN}" -c "import pybind11" >/dev/null 2>&1 || die "pybind11 is still missing after installing python/requirements.txt."
  "${PYTHON_BIN}" -c "import Cython" >/dev/null 2>&1 || die "Cython is still missing after installing python/requirements.txt."
}

echo "[check] source tree: ${SRC_HOME}"
check_python_env
check_dtk
check_aicc "${ROCM_PATH:-/opt/dtk}"
install_build_deps

function patchelf_version_ok() {
  local version
  version="$(patchelf --version 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)"
  [[ -n "${version}" ]] || return 1
  [[ "$(printf '%s\n%s\n' "0.14.5" "${version}" | sort -V | head -n1)" == "0.14.5" ]]
}

if [[ ${RUN_MANYLINUX} -eq 1 ]]; then
  if ! command -v auditwheel >/dev/null 2>&1; then
    echo "[check] auditwheel not found, installing it with ${PYTHON_BIN} -m pip"
    "${PYTHON_BIN}" -m pip install auditwheel || die "failed to install auditwheel. Pass --no-manylinux to skip the manylinux repair."
  fi
  command -v auditwheel >/dev/null 2>&1 || die "auditwheel is still not in PATH after installation. Pass --no-manylinux to skip the manylinux repair."
  if ! command -v patchelf >/dev/null 2>&1 || ! patchelf_version_ok; then
    echo "[check] patchelf missing or older than 0.14.5, installing it with ${PYTHON_BIN} -m pip"
    "${PYTHON_BIN}" -m pip install --upgrade "patchelf>=0.14.5" || die "failed to install patchelf >= 0.14.5."
    if ! command -v patchelf >/dev/null 2>&1 || ! patchelf_version_ok; then
      die "patchelf >= 0.14.5 is still not available in PATH after installation."
    fi
  fi
fi

if [[ -n "${LLVM_REPO}" && -z "${LLVM_SYSPATH:-}" && -z "${LLVM_BUILD_DIR:-}" ]]; then
  echo "[build] ensuring LLVM is available (local source build if the file server has no package)"
  # shellcheck disable=SC1091
  source "${CUR_PATH}/launch.sh" >/dev/null
  build_llvm "${LLVM_REPO}" "${FILE_SERVER_IP}" "${FILE_SERVER_USER}" "${FILE_SERVER_PWD}"
fi

function prepare_triton_cache() {
  local cache_dir="${TRITON_HOME:-${HOME}}/.triton"
  mkdir -p "${cache_dir}"
  (
    cd "${cache_dir}"
    for entry in "nvidia.3.6.x.tar.gz:nvidia" "json-v3.11.3.tar.gz:json"; do
      pkg="${entry%%:*}"
      dir="${entry##*:}"
      if [[ -d "${dir}" ]]; then
        continue
      fi
      if wget -q --timeout=10 --tries=1 "${TRITON_DEP_URL}/${pkg}" -O "${pkg}" && tar -zxf "${pkg}"; then
        echo "[build] prepared ${cache_dir}/${dir}"
      else
        rm -f "${pkg}"
        echo "[build] WARNING: could not prepare ${pkg}; setup.py will try to download it"
      fi
    done
  )
}

prepare_triton_cache

cd "${SRC_HOME}"
echo "[build] building wheel with ${PYTHON_BIN}"
if [[ -n "${LLVM_BUILD_DIR:-}" ]]; then
  LLVM_INCLUDE_DIRS="${LLVM_BUILD_DIR}/include" \
  LLVM_LIBRARY_DIR="${LLVM_BUILD_DIR}/lib" \
  LLVM_SYSPATH="${LLVM_BUILD_DIR}" \
  "${PYTHON_BIN}" setup.py bdist_wheel
else
  "${PYTHON_BIN}" setup.py bdist_wheel
fi

if [[ ${RUN_MANYLINUX} -eq 1 ]]; then
  echo "[build] repairing wheels for manylinux"
  bash "${CUR_PATH}/repair_manylinux.sh"
fi

shopt -s nullglob
wheels=(dist/*.whl)
shopt -u nullglob
[[ ${#wheels[@]} -gt 0 ]] || die "no wheel was produced under ${SRC_HOME}/dist"

echo -e "\nThe wheel path is:"
for wheel in "${wheels[@]}"; do
  echo "  $(realpath "${wheel}")"
done

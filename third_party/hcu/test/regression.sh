#!/bin/bash
set -euo pipefail

CUR_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CUR_PATH为脚本所在路径，进一步获取triton项目根路径
export SRC_HOME="$(realpath "${CUR_PATH}/../../../")"

hcu_file_path="${SRC_HOME}/third_party/hcu/test"
junit_xml="${JUNIT_XML:-${CUR_PATH}/hcu_regression.xml}"

pytest_cases=(
  "${hcu_file_path}/matmul.py"
  "${hcu_file_path}/rmsnorm.py"
  "${hcu_file_path}/fused-attention.py"
  "${hcu_file_path}/test_amd_buffer_ops_4gb.py"
  "${hcu_file_path}/test_amd_buffer_ops_offset_assert.py"
  "${hcu_file_path}/gluon/gluon_kernel_gemm_a8w8.py"
  "${hcu_file_path}/gluon/gluon_kernel_pa_decode.py"
  "${hcu_file_path}/gluon/gluon_kernel_pa_mqa_logits.py"
  "${hcu_file_path}/test_buffer_atomic_dtypes.py"
  "${hcu_file_path}/regression_subprocess.py"
)

pytest_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --junitxml=*)
      junit_xml="${1#--junitxml=}"
      ;;
    --junitxml)
      shift
      if [[ $# -eq 0 ]]; then
        echo "missing value for --junitxml" >&2
        exit 2
      fi
      junit_xml="$1"
      ;;
    *)
      pytest_args+=("$1")
      ;;
  esac
  shift
done

function run_pytest() {
  local junit_dir
  junit_dir="$(dirname "${junit_xml}")"
  mkdir -p "${junit_dir}"

  local pytest_cmd=()
  if [[ -n "${PYTEST:-}" ]]; then
    read -r -a pytest_cmd <<< "${PYTEST}"
  else
    pytest_cmd=("${PYTHON:-python}" -m pytest)
  fi

  echo "Run HCU pytest regression:"
  echo "  junitxml: ${junit_xml}"
  "${pytest_cmd[@]}" -s --tb=short "--junitxml=${junit_xml}" "${pytest_args[@]}" "${pytest_cases[@]}"
}

run_pytest

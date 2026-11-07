#!/bin/bash

set -euo pipefail

CUR_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CUR_PATH为脚本所在路径，进一步获取triton项目根路径
export SRC_HOME="$(realpath "${CUR_PATH}/../../../")"

hcu_file_path="${SRC_HOME}/third_party/hcu/test"
junit_xml="${JUNIT_XML:-${CUR_PATH}/hcu_regression.xml}"

base_pytest_cases=(
  "${hcu_file_path}/test_buffer_lds_config.py"
  "${hcu_file_path}/test_buffer_lds_copy.py"
  "${hcu_file_path}/test_async_copy_dtypes.py"
  "${hcu_file_path}/matmul.py"
  "${hcu_file_path}/rmsnorm.py"
  "${hcu_file_path}/fused-attention.py"
  "${hcu_file_path}/mls-unit-test.py"
  "${hcu_file_path}/mls-dot-scaled-test.py::test_mls_f8f6f4_comprehensive"
  "${hcu_file_path}/mls-dot-scaled-test.py::test_dot_accumulator_compile"
  "${hcu_file_path}/mls-dot-scaled-test.py::test_mls_dot_scaled_compile"
  "${hcu_file_path}/mls-dot-scaled-test.py::test_mls_with_scales_lowering"
  "${hcu_file_path}/mls-dot-scaled-test.py::test_mls_f8f6f4_scaled_values"
  "${hcu_file_path}/mls-dot-scaled-test.py::test_mls_fp4_atbt"
  "${hcu_file_path}/test_mls_async_pipeline.py"
  "${hcu_file_path}/test_mls_tiles_per_warp.py"
  "${hcu_file_path}/test_asyncmark_waitcnt.py"
  "${hcu_file_path}/gemm-ds-read-m.py"
  "${hcu_file_path}/test_elementwise_fma.py"
  "${hcu_file_path}/test_math_denorm.py"
  "${hcu_file_path}/test_cvt_scale_pk.py"
  "${hcu_file_path}/test_amd_buffer_ops_4gb.py"
  "${hcu_file_path}/test_amd_buffer_ops_offset_assert.py"
  "${hcu_file_path}/test_buffer_ops_branch_offset.py"
  "${hcu_file_path}/test_buffer_ops_mode_suite.py"
  "${hcu_file_path}/gluon/gluon_kernel_gemm_a8w8.py"
  "${hcu_file_path}/gluon/gluon_kernel_pa_decode.py"
  "${hcu_file_path}/gluon/gluon_kernel_pa_mqa_logits.py"
  "${hcu_file_path}/test_buffer_atomic_dtypes.py"
  "${hcu_file_path}/test_buffer_to_lds_gpu.py"
  "${hcu_file_path}/test_buffer_to_lds_passes.py"
  "${hcu_file_path}/test_buffer_publication_wait.py"
  "${hcu_file_path}/regression_subprocess.py"
  "${hcu_file_path}/test_hcu_barrier_pc_mlir.py"
  "${hcu_file_path}/test_misched_max_ilp_amdgcn.py"
  "${hcu_file_path}/test_amdgpu_max_memory_cluster_amdgcn.py"
  # WASP/WDRA IR shape: lit test/TritonGPU/hcu/hcu-gemm-wasp-wdra.mlir
  # Runtime matrix: test_wasp_mls_regression.py (run separately / via launch.sh)
  "${hcu_file_path}/test_mmac_instructions.py"
  "${hcu_file_path}/blocked-scale-matmul.py"
)

# Fixed-schedule performance references are checked only on the calibrated NMZ.
gemm_perf_cases=("${hcu_file_path}/test_gemm_performance.py")

# Compile-only checks stop at LLVM IR; no GPU or device libraries are needed.
scale_compile_file="${hcu_file_path}/test_scaled_compile.py"
scale_runtime_file="${hcu_file_path}/test_scaled_runtime.py"
scale_compile_cases=(
  "${scale_compile_file}::test_scaled_compile_paths"
  "${scale_compile_file}::test_scaled_compile_optional"
  "${scale_compile_file}::test_scaled_compile_wasp"
  "${scale_compile_file}::test_scaled_compile_wasp_mls_tile_reselection"
  "${scale_compile_file}::test_scaled_compile_formats"
  "${scale_compile_file}::test_scaled_compile_reuse"
  "${scale_compile_file}::test_scaled_compile_wasp_shares_matrix_barriers"
  "${scale_compile_file}::test_scaled_copy_m0_export"
  "${scale_compile_file}::test_scale_write_requires_lds"
  "${scale_compile_file}::test_scale_descriptor_owns_layout_and_base"
  "${scale_compile_file}::test_gluon_scaled_compile"
  "${scale_compile_file}::test_gluon_scaled_scalar_compile"
  "${scale_compile_file}::test_upstream_scaled_compile"
)

# Core numerical, synchronization and native-copy checks; the full runtime
# module also retains the larger bank-placement matrix for standalone runs.
scale_runtime_cases=(
  "${scale_runtime_file}::test_scaled_formats"
  "${scale_runtime_file}::test_two_stage_scale_pipeline"
  "${scale_runtime_file}::test_two_stage_mixed_scale_pipeline"
  "${scale_runtime_file}::test_scale_global_lds"
  "${scale_runtime_file}::test_optional_scale_pipeline"
  "${scale_runtime_file}::test_optional_scale_non_square"
  "${scale_runtime_file}::test_scaled_wasp"
  "${scale_runtime_file}::test_optional_scale_wasp"
  "${scale_runtime_file}::test_unscaled_mls_fragments"
  "${scale_runtime_file}::test_small_register_tile_is_correct"
  "${scale_runtime_file}::test_scale_regions_reuse_without_result_store"
  "${scale_runtime_file}::test_gfx946_global_lds_direct_copy"
  "${scale_runtime_file}::test_gluon_scaled_values"
  "${scale_runtime_file}::test_gluon_scaled_scalar_values"
)

suite="all"
pytest_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite=*)
      suite="${1#--suite=}"
      ;;
    --suite)
      shift
      if [[ $# -eq 0 ]]; then
        echo "missing value for --suite" >&2
        exit 2
      fi
      suite="$1"
      ;;
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

case "${suite}" in
  all)
    pytest_cases=("${base_pytest_cases[@]}" "${scale_compile_cases[@]}" "${scale_runtime_cases[@]}" "${gemm_perf_cases[@]}")
    ;;
  gemm-perf)
    pytest_cases=("${gemm_perf_cases[@]}")
    ;;
  scale)
    pytest_cases=("${scale_compile_cases[@]}" "${scale_runtime_cases[@]}")
    ;;
  scale-compile)
    pytest_cases=("${scale_compile_cases[@]}")
    ;;
  scale-runtime)
    pytest_cases=("${scale_runtime_cases[@]}")
    ;;
  *)
    echo "unknown suite: ${suite} (expected all, gemm-perf, scale, scale-compile or scale-runtime)" >&2
    exit 2
    ;;
esac

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

  echo "Run HCU pytest regression (${suite}):"
  echo "  junitxml: ${junit_xml}"
  "${pytest_cmd[@]}" -s --tb=short "--junitxml=${junit_xml}" "${pytest_args[@]}" "${pytest_cases[@]}"
}

run_pytest

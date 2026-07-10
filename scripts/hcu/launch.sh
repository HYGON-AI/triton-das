#!/bin/bash

CUR_PATH="$( cd $( dirname ${BASH_SOURCE} );pwd )"
export SRC_HOME=${CUR_PATH}/../../
export BUILD_DIR=${SRC_HOME}/build
export MAX_JOBS=8

function run_pytest() {
  test_files=(
    "test_performance_hcu.py"
  )
  file_path=${SRC_HOME}/python/test/regression
  for f in ${test_files[@]}; do
    pytest_file=${file_path}/${f}
    pytest ${pytest_file}
    ret=$?
    if [ $ret -ne 0 ]; then
      return $ret
    fi
  done
  echo -e "\n================="
  echo    "Run all passed!!!"
  echo -e "=================\n"
}

function run_google_test() {
  UNIT_TEST_DIR="${BUILD_DIR}/$(ls ${BUILD_DIR} | grep -i cmake)/unittest"
  if [ ! -d "${UNIT_TEST_DIR}" ]; then
    echo "Could not find '${UNIT_TEST_DIR}'"
    return -1
  fi

  cd ${UNIT_TEST_DIR}

  ALL_TESTS=`find . -type f -executable`
  test_exclude=(
    "TestPtxAsmFormat"
  )
  for test in ${ALL_TESTS[@]}; do
    is_exclude=False
    for exc in ${test_exclude[@]}; do
      if [[ $exc == ${test##*/} ]]; then
        is_exclude=True
        break
      fi
    done
    if [[ $is_exclude == "True" ]]; then
      continue
    fi
    ${test}
    ret=$?
    if [ $ret -ne 0 ]; then
      return $ret
    fi
  done
  cd -

  echo -e "\n================="
  echo    "Run all passed!!!"
  echo -e "=================\n"
}

function run_lit() {
  mode=$1
  if [[ $# -eq 0 || $mode -ne 0 && $mode -ne 1 ]]; then
    echo -e "\nError mode select for run_lit!"
    echo "Please run_lit with mode 0 or 1"
    echo "0: run all with exclude cases"
    echo "1: run only cases in test lists"
    echo "such as:"
    echo -e "$ run_lit 0\n"
    return -1
  fi
  LIT_TEST_DIR="${BUILD_DIR}/$(ls ${BUILD_DIR} | grep -i cmake)/test"
  if [ ! -d "${LIT_TEST_DIR}" ]; then
    echo "Could not find '${LIT_TEST_DIR}'"
    return -1
  fi

  test_list=(
    "${LIT_TEST_DIR}/Analysis/test-alias.mlir"
    "${LIT_TEST_DIR}/Analysis/test-alignment.mlir"
    "${LIT_TEST_DIR}/Analysis/test-allocation.mlir"
    "${LIT_TEST_DIR}/Analysis/test-membar.mlir"
    "${LIT_TEST_DIR}/Conversion/AMDGPU/load_store.mlir"
    "${LIT_TEST_DIR}/Conversion/dedup-by-constancy.mlir"
    "${LIT_TEST_DIR}/Conversion/invalid.mlir"
    "${LIT_TEST_DIR}/Conversion/triton_ops.mlir"
    "${LIT_TEST_DIR}/Conversion/triton_to_tritongpu.mlir"
    "${LIT_TEST_DIR}/LLVMIR/break-phi-struct.ll"
    "${LIT_TEST_DIR}/Triton/canonicalize.mlir"
    "${LIT_TEST_DIR}/Triton/combine.mlir"
    "${LIT_TEST_DIR}/Triton/print.mlir"
    "${LIT_TEST_DIR}/Triton/reorder-broadcast.mlir"
    "${LIT_TEST_DIR}/Triton/vecadd.mlir"
    "${LIT_TEST_DIR}/TritonGPU/accelerate-amd-matmul.mlir"
    "${LIT_TEST_DIR}/TritonGPU/accelerate-matmul.mlir"
    "${LIT_TEST_DIR}/TritonGPU/atomic-cas.mlir"
    "${LIT_TEST_DIR}/TritonGPU/canonicalize.mlir"
    "${LIT_TEST_DIR}/TritonGPU/chain-dot.mlir"
    "${LIT_TEST_DIR}/TritonGPU/coalesce.mlir"
    "${LIT_TEST_DIR}/TritonGPU/combine.mlir"
    "${LIT_TEST_DIR}/TritonGPU/dot-operands.mlir"
    "${LIT_TEST_DIR}/TritonGPU/dot-slicing.mlir"
    "${LIT_TEST_DIR}/TritonGPU/fence-inserstion.mlir"
    "${LIT_TEST_DIR}/TritonGPU/loop-pipeline-hopper.mlir"
    "${LIT_TEST_DIR}/TritonGPU/loop-pipeline.mlir"
    "${LIT_TEST_DIR}/TritonGPU/materialize-load-store.mlir"
    "${LIT_TEST_DIR}/TritonGPU/matmul.mlir"
    "${LIT_TEST_DIR}/TritonGPU/optimize-locality.mlir"
    "${LIT_TEST_DIR}/TritonGPU/pipeline-hopper-remove-wait.mlir"
    "${LIT_TEST_DIR}/TritonGPU/prefetch.mlir"
    "${LIT_TEST_DIR}/TritonGPU/remove-layout-conversions.mlir"
    "${LIT_TEST_DIR}/TritonGPU/reorder-instructions.mlir"
    "${LIT_TEST_DIR}/TritonGPU/rewrite-tensor-pointer-tma.mlir"
    "${LIT_TEST_DIR}/TritonGPU/rewrite-tensor-pointer.mlir"
    "${LIT_TEST_DIR}/TritonGPU/stream-pipeline.mlir"
    "${LIT_TEST_DIR}/TritonGPU/wsmaterialization.mlir"
    "${LIT_TEST_DIR}/TritonGPU/wsmutex.mlir"
    "${LIT_TEST_DIR}/TritonGPU/wspipeline.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/hcu-gemm-wasp-wdra.mlir"
  )
  test_exclude=(
    "${LIT_TEST_DIR}/Conversion/AMDGPU/mfma_variants.mlir"
    "${LIT_TEST_DIR}/Conversion/divide-by-0.mlir"
    "${LIT_TEST_DIR}/Conversion/minimize_alloc.mlir"
    "${LIT_TEST_DIR}/Conversion/tritongpu_to_llvm.mlir"
    "${LIT_TEST_DIR}/Conversion/tma_to_llvm.mlir"
    "${LIT_TEST_DIR}/Conversion/cvt_to_llvm.mlir"
    "${LIT_TEST_DIR}/Conversion/AMDGPU/mfma-shortcut.mlir"
    "${LIT_TEST_DIR}/Conversion/AMDGPU/ds_transpose.mlir"
    "${LIT_TEST_DIR}/TritonGPU/accelerate-matmul-cdna1.mlir"
    "${LIT_TEST_DIR}/TritonGPU/accelerate-matmul-cdna2.mlir"
    "${LIT_TEST_DIR}/TritonGPU/accelerate-matmul-cdna3.mlir"
    "${LIT_TEST_DIR}/TritonGPU/chain-dot.mlir"
    "${LIT_TEST_DIR}/TritonGPU/prefetch.mlir"
    "${LIT_TEST_DIR}/TritonGPU/combine.mlir"
    "${LIT_TEST_DIR}/TritonGPU/loop-pipeline-hip.mlir"
    "${LIT_TEST_DIR}/Conversion/tritongpu_to_llvm_hopper.mlir"
    "${LIT_TEST_DIR}/Conversion/AMDGPU/math-denorm-handling.mlir"
    "${LIT_TEST_DIR}/Conversion/AMDGPU/fp_to_fp.mlir"
    "${LIT_TEST_DIR}/TritonGPU/wsdecomposing.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/accelerate-amd-matmul-mfma.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/mfma-xf32.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/sink-setprio-mfma.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/amd-optimize-epilogue.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/accelerate-amd-matmul-mfma-gfx950.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/optimize-lds-usage.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/amd-reorder-instructions.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/amd-block-pingpong.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/amd-optimize-dot-operands.mlir"
    "${LIT_TEST_DIR}/TritonGPU/amd/mfma-double-rate.mlir"
    "${LIT_TEST_DIR}/Tools/tensor_layout_print.mlir"
  )
  # run all with exclude
  if [ $mode -eq 0 ]; then
    filter=${test_exclude[0]##*/}
    for exclude in ${test_exclude[@]}; do
      exc_case=${exclude##*/}
      if [[ $exc_case == $filter ]]; then
        continue
      fi
      filter="${filter}|${exc_case}"
    done
    lit -v ${LIT_TEST_DIR} --filter-out=${filter}
    ret=$?
    if [ $ret -ne 0 ]; then
      return $ret
    fi
  elif [ $mode -eq 1 ]; then
    # only run test list
    for case in ${test_list[@]}; do
      echo "case:"
      echo "${case}"
      lit -v ${case}
      ret=$?
      if [ $ret -ne 0 ]; then
        return $ret
      fi
    done
  fi
  echo -e "\n================="
  echo    "Run all passed!!!"
  echo -e "=================\n"
}

function clean_cache() {
  cd ${SRC_HOME}
  rm -rf python/triton.egg-info
  rm -rf python/.pytest_cache
  rm -rf python/tests/__pycache__
  rm -rf python/build
  rm -rf /root/.triton/cache
  rm -rf /tmp/*
  rm -rf triton_cache
  cd -
}

function build_llvm() {
  if [ $# -ne 4 ]; then
    echo "Error: 4 arguments required"
    return -1
  fi
  # Use default value for function arguments
  repo=${1:-"ssh://git@10.65.42.70:8022/buhui/llvm-project.git"}
  package_server=${2:-"10.65.42.71"}
  package_user=${3:-"sw-builder"}
  password=${4:-"swadmin"}
  pushd ${SRC_HOME}
  package_path="/public/opendas/ArchivedFile/Jenkins/CompileDep/triton"
  hash=`grep -m 1 -v '^$' cmake/llvm-hash.txt`
  hash=${hash:0:8}
  platform=$(cat /etc/*release | grep '^ID=' | awk -F '=' '{print $2}' | tr -d '"')
  package_name=llvm-${hash}-${platform}-x64
  # Package not exist, build the package
  set +e
  curl --head --silent --fail http://42.228.13.241:18000/Jenkins/CompileDep/triton/${package_name}.tar.gz
  check_exist=$?
  set -e
  if [[ $check_exist -ne 0 ]]; then
    # git repo already exist
    if [ -d "llvm-project" ] && [ -d "llvm-project/.git" ]; then
      cd llvm-project
      git fetch
    # git repo not exist, reclone
    else
      rm -rf llvm-project
      git clone ${repo} llvm-project
      cd llvm-project
    fi
    git reset ${hash} --hard
    git clean -fd
    mkdir -p build
    mkdir -p ${package_name}
    rm -rf ${package_name} ${package_name}.tar.gz
    cd build
    cmake -G Ninja  \
      -DCMAKE_BUILD_TYPE=Release  \
      -DLLVM_ENABLE_ASSERTIONS=ON  \
      -DLLVM_INSTALL_UTILS=ON  \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld"  \
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"  \
      -DCMAKE_INSTALL_PREFIX=../${package_name}  \
      ../llvm
    ninja -j${MAX_JOBS}
    ninja install -j${MAX_JOBS}
    cd ..
    tar -zcf ${package_name}.tar.gz ${package_name}
    export RSYNC_PWD="${password}"
    expect -c """
      set timeout 14400
      spawn rsync -avP ${package_name}.tar.gz ${package_user}@${package_server}:${package_path}/
      expect {
        \"*yes/no*\" {
          send \"yes\n\"
            expect \"*assword:\" {
            send \"\$env(RSYNC_PWD)\n\"
            expect eof
          }
        }
        \"*assword:\" {
          send \"\$env(RSYNC_PWD)\n\"
          expect eof
        }
        eof
      }
      wait
    """
  fi
  popd
}

function usage() {
  echo "Usage:"
  echo "Firstly:"
  echo "$ source launch.sh"
  echo "Then:"
  echo "============================"
  echo "Run googletest:"
  echo "$ run_google_test"
  echo "============================"
  echo "Run lit:"
  echo "$ run_lit 0"
  echo "or"
  echo "$ run_lit 1"
  echo "============================"
  echo "Run pytest:"
  echo "$ run_pytest"
  echo "============================"
  echo -e "\n==================="
  echo    "Launch Successed!!!"
  echo -e "===================\n"
}

usage

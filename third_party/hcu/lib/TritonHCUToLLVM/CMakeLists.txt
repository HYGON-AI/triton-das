# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set(TRITON_HCU_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/amd/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../amd/include
    )

add_triton_library(TritonHCUToLLVM
  TritonHCUToLLVM.cpp
  DotOpToLLVM/MMAC.cpp
  DotOpToLLVMHCU.cpp
  ElementwiseOpToLLVM.cpp
  DsReadM.cpp
  MlsOpToLLVM.cpp
  WaitCntHCUUtility.cpp
  ConvertWarpSpecializeToLLVM.cpp

  DEPENDS
  TritonHCUPassesIncGen
  TritonAMDGPUConversionPassIncGen
  LLVMIRIncGen

  LINK_LIBS PUBLIC
  TritonHCUTransforms
  TritonGPUToLLVM
  TritonAMDGPUIR
  LLVMCore
  LLVMPasses
  LLVMSupport
)

target_include_directories(TritonHCUToLLVM PRIVATE ${TRITON_HCU_INCLUDE_DIRS})
target_include_directories(TritonHCUToLLVM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/amd/lib/TritonAMDGPUToLLVM)
target_include_directories(TritonHCUToLLVM PRIVATE ${PROJECT_BINARY_DIR}/third_party/amd/lib/TritonAMDGPUToLLVM)

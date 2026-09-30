set(TRITON_HCU1_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/amd/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../amd/include
)

# Light helpers used by core DataPartition. Must NOT pull Accelerate /
# AMDGPUToLLVM, or PUBLIC consumers (googletest) fail to link.
# Keep AMDGPUIR PRIVATE so googletest is not forced to resolve AMDGPUIR's
# own incomplete link deps (Utils / GPUToLLVM linearize).
add_triton_library(TritonHCUWaspSupport
  MlsGroup.cpp
  WaspSplitPlan.cpp

  DEPENDS
  TritonAMDGPUIR
  TritonGPUIR

  LINK_LIBS PUBLIC
  TritonGPUIR
)

target_include_directories(TritonHCUWaspSupport PRIVATE ${TRITON_HCU1_INCLUDE_DIRS})
target_link_libraries(TritonHCUWaspSupport PRIVATE TritonAMDGPUIR)

add_triton_library(TritonHCUTransforms
  AccelerateHCUMatmul.cpp
  BufferLdsEncoding.cpp
  SelectBufferLdsConfig.cpp
  ConvertBufferLdsCopies.cpp
  ApplyFuncHintsAndClean.cpp
  ConsumerPingpong.cpp
  BlockPingpong.cpp
  UTCWarmup.cpp
  MlsEncodingInsertion.cpp
  MlsStreamPipeline.cpp
  MlsLowering.cpp
  UpdateAsyncWaitCount.cpp
  MmacGroup.cpp
  PrepareWaspSplitPlan.cpp
  UtilityHCU.cpp

  DEPENDS
  TritonHCUPassesIncGen
  TritonAMDGPUIR
  TritonAMDGPUTransformsIncGen
  TritonGPUIR
  TritonHCUWaspSupport

  LINK_LIBS PUBLIC
  TritonHCUWaspSupport
  TritonAMDGPUIR
  TritonAMDGPUTransforms
  TritonAMDGPUToLLVM
  TritonGPUIR
  TritonAMDUtils
  TritonAMDAnalysis
  MLIRLLVMDialect
)

target_include_directories(TritonHCUTransforms PRIVATE ${TRITON_HCU1_INCLUDE_DIRS})

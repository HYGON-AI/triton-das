set(TRITON_HCU1_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/amd/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../amd/include
)

# Light helpers used by core DataPartition. Must NOT pull Accelerate /
# AMDGPUToLLVM, or PUBLIC consumers (googletest) fail to link.
# Keep AMDGPUIR PRIVATE so googletest is not forced to resolve AMDGPUIR's
# own incomplete link deps (Utils / GPUToLLVM linearize).
add_triton_library(TritonHCUWdraSupport
  MlsGroup.cpp
  WdraSplitPlan.cpp

  DEPENDS
  TritonAMDGPUIR
  TritonGPUIR

  LINK_LIBS PUBLIC
  TritonGPUIR
)

target_include_directories(TritonHCUWdraSupport PRIVATE ${TRITON_HCU1_INCLUDE_DIRS})
target_link_libraries(TritonHCUWdraSupport PRIVATE TritonAMDGPUIR)

add_triton_library(TritonHCUTransforms
  AccelerateHCUMatmul.cpp
  MlsEncodingInsertion.cpp
  MlsStreamPipeline.cpp
  MlsLowering.cpp
  MmacGroup.cpp
  PrepareWdraSplitPlan.cpp
  UtilityHCU.cpp

  DEPENDS
  TritonHCUPassesIncGen
  TritonAMDGPUIR
  TritonAMDGPUTransformsIncGen
  TritonGPUIR
  TritonHCUWdraSupport

  LINK_LIBS PUBLIC
  TritonHCUWdraSupport
  TritonAMDGPUIR
  TritonGPUIR
  TritonAMDUtils
  TritonAMDAnalysis
)

target_include_directories(TritonHCUTransforms PRIVATE ${TRITON_HCU1_INCLUDE_DIRS})

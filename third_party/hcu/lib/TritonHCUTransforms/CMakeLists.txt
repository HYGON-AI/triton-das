set(TRITON_HCU1_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/amd/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../amd/include
)

add_triton_library(TritonHCUTransforms
  AccelerateHCUMatmul.cpp
  MlsGroup.cpp
  MlsEncodingInsertion.cpp
  MlsStreamPipeline.cpp
  MlsLowering.cpp
  MmacGroup.cpp
  UtilityHCU.cpp

  DEPENDS
  TritonHCUPassesIncGen
  TritonAMDGPUIR
  TritonAMDGPUTransformsIncGen
  TritonGPUIR

  LINK_LIBS PUBLIC
  TritonAMDGPUIR
  TritonGPUIR
  TritonAMDUtils
  TritonAMDAnalysis
)

target_include_directories(TritonHCUTransforms PRIVATE ${TRITON_HCU1_INCLUDE_DIRS})

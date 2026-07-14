# HCU backend — Hygon DCU-specific passes, lowering, and pybind11 module.
# Depends on AMD shared libraries (TritonAMDGPUToLLVM, TritonAMDGPUTransforms, etc.).

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/amd/include)
include_directories(${PROJECT_BINARY_DIR}/third_party/amd/include)
add_subdirectory(include)

# Compile shared AMD libraries if not already compiled by the amd backend.
# This allows standalone builds with CODEGEN_BACKENDS=hcu.
if(NOT TARGET TritonAMDGPUToLLVM)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/amd/lib/TritonAMDGPUToLLVM third_party/amd/lib/TritonAMDGPUToLLVM)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/amd/lib/TritonAMDGPUTransforms third_party/amd/lib/TritonAMDGPUTransforms)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/amd/lib/TritonAMDGPUDialectToLLVM third_party/amd/lib/TritonAMDGPUDialectToLLVM)
endif()

add_subdirectory(lib)

# DataPartition (core TritonGPUTransforms) reselects MLS tiles for WDRA slices.
if(TARGET TritonGPUTransforms)
  target_link_libraries(TritonGPUTransforms PUBLIC TritonHCUTransforms)
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
  include_directories(${LLD_INCLUDE_DIRS})

  add_triton_plugin(TritonHCU
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_hcu.cc
    LINK_LIBS
    TritonHCUToLLVM
    TritonHCUTransforms
    TritonAMDGPUToLLVM
    TritonAMDGPUTransforms
    TritonAMDGPUDialectToLLVM
  )
  target_link_libraries(TritonHCU PRIVATE Python3::Module pybind11::headers lldCommon lldELF)
endif()

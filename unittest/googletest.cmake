# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.

include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

FetchContent_Declare(
  googletest
  # Use repo hosted on gitee instead for a better network connection.
  # GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_REPOSITORY https://gitee.com/mirrors/googletest.git
  GIT_TAG v1.17.0
  )

FetchContent_GetProperties(googletest)

if(NOT googletest_POPULATED)
  FetchContent_MakeAvailable(googletest)
  if (MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
endif()

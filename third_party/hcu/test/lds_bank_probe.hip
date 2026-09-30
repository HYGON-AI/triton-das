// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef READ_INSTRUCTION
#define READ_INSTRUCTION "ds_read2_b32"
#endif
#ifndef READ_SUFFIX
#define READ_SUFFIX "offset0:0 offset1:32"
#endif
#ifndef RESULT_WORDS
#define RESULT_WORDS 2
#endif
#ifndef EXPECT_ARCH
#define EXPECT_ARCH "gfx936"
#endif
using Registers = unsigned __attribute__((ext_vector_type(RESULT_WORDS)));

__global__ void lds_bank_probe(unsigned *out, unsigned stride,
                               unsigned first, unsigned second, unsigned collide) {
  __shared__ unsigned words[8192];
  unsigned lane = threadIdx.x;
  for (unsigned i = lane; i < 8192; i += 64)
    words[i] = i;
  __syncthreads();
  unsigned address = unsigned(reinterpret_cast<uintptr_t>(words)) + lane * stride;
  if (first < 64)
    address = unsigned(reinterpret_cast<uintptr_t>(words)) +
              (lane == first ? 16 : lane == second ? (collide ? 144 : 32) : 0);
  Registers values;
  // Serial issue makes the tested opcode observable and prevents load CSE or
  // read2 combining from changing the instruction under test.
  for (unsigned i = 0; i < 1024; ++i)
    asm volatile(READ_INSTRUCTION " %0, %1 " READ_SUFFIX
                 "\n\ts_waitcnt lgkmcnt(0)"
                 : "=v"(values) : "v"(address) : "memory");
  for (unsigned reg = 0; reg < RESULT_WORDS; ++reg)
    out[lane * RESULT_WORDS + reg] = values[reg];
  return;
}
static void check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(status));
    std::exit(1);
  }
}
int main(int argc, char **argv) {
  hipDeviceProp_t properties;
  check(hipGetDeviceProperties(&properties, 0));
  if (std::strncmp(properties.gcnArchName, EXPECT_ARCH,
                   std::strlen(EXPECT_ARCH)) != 0)
    return 2;
  unsigned stride = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 4;
  if (stride > 128)
    return 2;
  unsigned *out;
  check(hipMalloc(&out, 64 * RESULT_WORDS * sizeof(unsigned)));
  unsigned first = argc > 2 ? std::strtoul(argv[2], nullptr, 0) : 64;
  unsigned second = argc > 3 ? std::strtoul(argv[3], nullptr, 0) : 64;
  unsigned collide = argc > 4 ? std::strtoul(argv[4], nullptr, 0) : 0;
  lds_bank_probe<<<1, 64>>>(out, stride, first, second, collide);
  check(hipGetLastError());
  check(hipDeviceSynchronize());
  check(hipFree(out));
  return 0;
}

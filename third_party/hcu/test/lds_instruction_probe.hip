// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Isolated source-bit/register mapping probe. Compile one opcode at a time;
// never run an opcode merely because an assembler accepts its encoding.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef READ_INSTRUCTION
#define READ_INSTRUCTION "ds_read_m32x16_b16"
#endif
#ifndef ELEMENT_BITS
#define ELEMENT_BITS 16
#endif
#ifndef EXPECT_ARCH
#define EXPECT_ARCH "gfx936"
#endif
using Registers = unsigned __attribute__((ext_vector_type(4)));

__global__ void read_mapping(unsigned *out, unsigned digit) {
  __shared__ unsigned words[256];
  unsigned lane = threadIdx.x;
  constexpr unsigned perWord = 32 / ELEMENT_BITS;
  constexpr unsigned mask = unsigned((uint64_t(1) << ELEMENT_BITS) - 1);
  for (unsigned word = lane; word < 256; word += 64) {
    unsigned packed = 0;
    for (unsigned item = 0; item < perWord; ++item)
      packed |= (((word * perWord + item) >> digit) & mask)
                << (item * ELEMENT_BITS);
    words[word] = packed;
  }
  __syncthreads();
  unsigned address = static_cast<unsigned>(reinterpret_cast<uintptr_t>(words)) + lane * 16;
  Registers values;
  asm volatile(READ_INSTRUCTION " %0, %1\n\ts_waitcnt lgkmcnt(0)"
               : "=v"(values) : "v"(address) : "memory");
  for (unsigned reg = 0; reg < 4; ++reg)
    out[lane * 4 + reg] = values[reg];
  return;
}

static void check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(status));
    std::exit(1);
  }
}
int main() {
  hipDeviceProp_t properties;
  check(hipGetDeviceProperties(&properties, 0));
  if (std::strncmp(properties.gcnArchName, EXPECT_ARCH,
                   std::strlen(EXPECT_ARCH)) != 0) {
    std::fprintf(stderr, "This probe's execution gate requires %s\n",
                 EXPECT_ARCH);
    return 2;
  }
  unsigned *device = nullptr;
  unsigned host[256];
  check(hipMalloc(&device, sizeof(host)));
  // Multiple digits distinguish every source element even for b4/b8.
  for (unsigned digit = 0; digit < 11; digit += ELEMENT_BITS) {
    read_mapping<<<1, 64>>>(device, digit);
    check(hipGetLastError());
    check(hipMemcpy(host, device, sizeof(host), hipMemcpyDeviceToHost));
    for (unsigned lane = 0; lane < 64; ++lane)
      std::printf("%u,%u,%u,%u,%u,%u\n", digit, lane, host[lane * 4],
                  host[lane * 4 + 1], host[lane * 4 + 2], host[lane * 4 + 3]);
  }
  check(hipFree(device));
}

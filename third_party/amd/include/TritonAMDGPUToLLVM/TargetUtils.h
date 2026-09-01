// Modified by Hygon Information Technology Co., Ltd., 2026.

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_TARGETUTILS_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_TARGETUTILS_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::AMD {

// A list of ISA families we care about.
enum class ISAFamily {
  Unknown,
  CDNA1,
  CDNA2,
  CDNA3,
  CDNA4,
  RDNA1,
  RDNA2,
  RDNA3,
  RDNA4,
  GFX1250,
};

// Deduces the corresponding ISA family for the given target gfx |arch|.
ISAFamily deduceISAFamily(llvm::StringRef arch);

// Retursn true if given architecture support V_DOT instruction.
bool supportsVDot(llvm::StringRef arch);

bool isCDNA(ISAFamily isaFamily);

bool isRDNA(ISAFamily isaFamily);

// Here is a partial definition of DppCtrl enums. For the complete definition,
// please check:
// https://github.com/llvm/llvm-project/blob/8c75290/llvm/lib/Target/AMDGPU/SIDefines.h#L939
enum class DppCtrl : uint32_t {
  QUAD_PERM_FIRST = 0,
  ROW_SHL0 = 0x100,
  ROW_SHR0 = 0x110,
  BCAST15 = 0x142,
  BCAST31 = 0x143
};

// ISA capability features shared between AMD and HCU.
// The enum type lives here so that AMD public headers (e.g. MfmaGroup.h)
// can use it in function signatures without depending on HCU headers.
enum class HCUISAFeature : uint64_t {
  NONE          = 0,
  MMAC_LAYOUT   = 1 << 0,
  MAMC_FP8      = 1 << 1,
  MMAC_FP6FP4   = 1 << 2,
  MMAC_SCALE    = 1 << 3,
  MLS           = 1 << 4,
  MLS_FP6FP4    = MMAC_FP6FP4,
  MLS_B4        = MMAC_SCALE,
  CVT_FP8F32    = 1 << 5,
  CVT_SCALE_PK  = 1 << 6, // gfx946 V_CVT_SCALE_PK_* (E8M0); prefer over CVT_FP8F32
  MMAC_ACC_FP16 = 1 << 7,
  MMAC_ACC_BF16 = 1 << 8,
  MMAC_F32_K8   = 1 << 9, // V_MMAC_16X16X8_F32 (f32 x f32); only gfx928/936/938. Other HCUs use 16x16x4_F32.
};

inline constexpr HCUISAFeature operator~(HCUISAFeature feature) {
  return static_cast<HCUISAFeature>(~static_cast<uint64_t>(feature));
}
struct HCUISAFeatureMask {
  uint64_t bits;
  operator bool() const { return bits != 0; }
  HCUISAFeature operator|(HCUISAFeature rhs) const {
    return static_cast<HCUISAFeature>(bits | static_cast<uint64_t>(rhs));
  }
};
inline constexpr HCUISAFeatureMask operator&(HCUISAFeature lhs, HCUISAFeature rhs) {
  return {static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs)};
}
inline constexpr HCUISAFeature operator|(HCUISAFeature lhs, HCUISAFeature rhs) {
  return static_cast<HCUISAFeature>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}
inline constexpr HCUISAFeature &operator&=(HCUISAFeature &lhs, HCUISAFeature rhs) {
  lhs = static_cast<HCUISAFeature>(static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
  return lhs;
}

HCUISAFeature deduceHCUISAFeature(llvm::StringRef arch);
bool supportsHCUISAFeature(llvm::StringRef arch, HCUISAFeature feature);

// Buffer cache swizzle: kongming (gfx926), zhongda (gfx928), bmz (gfx936),
// nmz (gfx938), yueying (gfx92a), shaobo (gfx946). Ordinary inference remains
// capped at 8 KiB; target-specific callers may request extended stride bits.
bool supportsBufferCacheSwizzle(llvm::StringRef arch);

// NMZ and ShaoBo carry cache-swizzle stride high bits in the resource
// descriptor and can encode the UTC warmup 32 KiB policy. Yueying remains on
// the legacy 8 KiB policy.
bool supports32KiBCacheSwizzle(llvm::StringRef arch);

// Round stride up to a legal 14-bit swizzle value (64*2^n, max 16383). If
// minStrideBytes > 0 (matrix row pitch), the result is at least that value.
// Normalized stride is capped at 8192 bytes (8KB).
int64_t normalizeCacheSwizzleStrideBytes(int64_t strideBytes,
                                         int64_t minStrideBytes = 0);

// Return the LLVM intrinsic name for permlane swap on the given arch.
// gfx946 uses HCU builtins; CDNA4 (gfx950) uses amdgcn intrinsics.
llvm::StringRef getPermlane16SwapIntrinsic(llvm::StringRef arch);
llvm::StringRef getPermlane32SwapIntrinsic(llvm::StringRef arch);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_TARGETUTILS_H_

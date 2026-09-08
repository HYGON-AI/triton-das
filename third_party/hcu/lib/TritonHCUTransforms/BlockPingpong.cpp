#include "TritonHCU/DsReadMLayout.h"
#include "TritonHCU/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <limits>
#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonhcu-block-pingpong"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace tta = mlir::triton::amdgpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUPREPAREBLOCKPINGPONG
#define GEN_PASS_DEF_TRITONHCUBLOCKPINGPONG
#define GEN_PASS_DEF_TRITONHCUFINALIZEBLOCKPINGPONG
#define GEN_PASS_DEF_TRITONHCUPACKBLOCKPINGPONGBIT8
#include "TritonHCU/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// HCU BufferLoad-only BlockPingpong
//===----------------------------------------------------------------------===//
//
// This is a throughput optimization for large GEMMs and GEMM-like operators,
// including scaled GEMMs and the GEMM reductions inside fused MoE kernels.  It
// is not keyed to frontend operator names: any sufficiently large regular
// tt.dot reduction may use the schedule when its A/B dataflow, tile, load
// vectorization, MFMA shape, and LDS footprint satisfy the legality contract
// below.  Small or unsupported reductions are deliberately left to the normal
// AMD pipeline and autotuning remains responsible for choosing PP versus
// non-PP for each workload.
//
// The three TTGIR HCU passes in this file cooperate with the generic AMD
// pipeline; none of them is a second, independent software pipeliner:
//
//   TritonHCUPrepareBlockPingpong
//     -> AMD ScheduleLoops
//     -> AMD Pipeline
//     -> normal TTGIR cleanup/reordering
//     -> TritonHCUBlockPingpong
//     -> Triton LoopUnroll
//     -> Canonicalizer
//     -> TritonHCUFinalizeBlockPingpong
//
// A fourth, narrowly scoped LLVM cleanup pass packs Bit8 loop-carried values
// only in functions where the TTGIR scheduler actually applied this family.
//
// Prepare runs on the original tt.load -> tt.dot loop.  It performs the strict
// legality match and supplies the generic AMD pipeline with num_stages=3,
// local-store stage 1, and two LDS buffers.  These per-loop overrides do not
// depend on the frontend num_stages value.  AMD ScheduleLoops/Pipeline then
// creates the loop-carried global prefetch values, local_store/local_load ops,
// combined two-slot allocations, and the ordinary prologue/epilogue.
// Loop-invariant masks, such as MoE token-validity masks, remain on the normal
// pipeline.  When a loop-variant mask is proven to describe one K-boundary
// remainder, Prepare splits an unmasked main loop from a one-shot masked tail;
// only the main loop receives the BlockPingpong pipeline contract.
//
// BlockPingpong runs only after that expansion.  It revalidates the
// materialized loop and selects either the BK/2 M0-D0-M1-D1 schedule or the
// whole-BK M-D schedule from the operand width and reduction dataflow.  It then
// inserts asymmetric synchronization/priority boundaries and requests late
// unroll=2. The Python add_block_pingpong wrapper deliberately bundles this
// pass with LoopUnroll and Canonicalizer so two consecutive BK iterations
// expose fixed slot1/slot0 parity.
//
// Finalize consumes that static parity.  It replaces each combined [2, ...]
// allocation with explicit physical slot allocations, allowing Allocation and
// Membar analysis to reason about current-slot reads and next-slot writes.
// Normal operands receive independent slot0/slot1 allocations; a proven
// single-buffer operand maps both generations to one allocation and carries
// explicit local barriers on both sides of its overwrite. Prepare and
// BlockPingpong leave unmatched loops alone.  Once BlockPingpong marks a main
// loop as applied, however, Finalize treats static-buffer materialization as
// required: silently retaining an aliased [2,
// ...] buffer would invalidate the performance contract with Membar analysis.
//
// This file deliberately does not implement MLS or direct-to-LDS async copy.
// It owns one measured HCU schedule family for regular tt.dot reductions:
//
//   BM/BN >= 128, 128-aligned, legal BK half-tiles,
//   8 waves, num_stages=3
//   three functional levels + two logical LDS generations + two/four PP phases
//   + 2K unroll
//
// The pipeline-stage count, LDS-buffer count, PP-phase count, and unroll factor
// describe independent concepts.  The three stages perform global load,
// local store, and local load+dot respectively; unlike the former
// stage4/store2 schedule, there is no empty register-prefetch stage extending
// the loop-carried VGPR live range.  num_stages=3 still uses only two LDS
// buffers.  One generation is A(BM x BK) + B(BK x BN).  The 16-bit family
// (FP16/BF16) uses its existing legal BK values; the 8-bit family (FP8/i8/u8)
// normally uses BK64 so each BK/2 phase contains a complete 8-bit MMAC K tile.
// BM128 x BN128 x BK128 retains both A/B rings and uses the split four-phase
// path for a directly yielded dot; the asymmetric BK128 extensions reuse their
// larger operand physically and load that complete operand before its
// overwrite barrier.  The same physical geometry applies to the asymmetric
// Bit16 BK64 tiles. Bit8 BK128 additionally requires a K-divisible main loop;
// its one-shot masked-tail form falls back to the ordinary pipeline. All of
// these extended layouts use 64 KiB. The
// matcher proves that the final physical allocation contract fits the HCU LDS
// budget before selecting either schedule.  Initial legality is independent of
// concrete accumulator and epilogue op names: FP16/BF16, FP8/i8/u8,
// channel-wise scaling, bias, activation, and MoE routing may share this
// schedule when they lower to one regular dot per BK and only the A/B tiles
// enter LDS.  The phase
// selector later follows the unique dot-dependent loop yield to distinguish a
// direct accumulator from a separate post-dot update.  Auxiliary loop loads
// (for example one scale shared by the complete BK tile) may remain in the
// ordinary register pipeline.  Multiple dots per BK, operand dequantization
// before the dot, and tt.dot_scaled need their own slicing contract and are
// conservatively left unchanged.
//
// "BPP8" in tuning results means this required 8-wave schedule; it must not be
// confused with "Bit8", which describes FP8/i8/u8 operand storage.  The
// measured 8-wave tile families intentionally have different LDS contracts:
//
//   BM256 x BN256 x BK32 Bit16 / BK64 Bit8
//     The original square family.  A and B both keep two physical slots, using
//     64 KiB LDS.
//
//   BM128 x BN256 or BM256 x BN128, BK32 Bit16 / BK64 Bit8
//     Asymmetric output tiles in the same original family.  A and B still keep
//     two physical slots; the LDS footprint falls to 48 KiB.  These tiles are
//     separate autotune candidates, not aliases for the square schedule.
//
//   BM256 x BN128 x BK128 Bit8
//     The measured MoE extension.  Its two complete A+B generations would use
//     96 KiB, so Finalize maps both logical A generations to one 32 KiB
//     physical allocation while retaining two 16 KiB B allocations.  Like
//     the transposed single-B path, M0 reads the complete physical-single
//     operand and D0 ends at a full-CTA local barrier before M1 may overwrite
//     it.  Safety therefore follows from read-before-overwrite synchronization
//     and does not depend on a particular MMAC layout enumeration.
//
//   BM128 x BN128 x BK128 Bit8
//     A and B each retain two physical 16 KiB slots, using exactly 64 KiB.
//     Its directly yielded dot uses the common split-BK four-phase topology.
//     No overwrite barrier is needed because current and next generations
//     never share a physical slot.
//
//   BM128 x BN256 x BK128 Bit8
//     The mirrored MoE extension.  Two A slots plus two B slots would use
//     96 KiB, so Finalize retains two 16 KiB A allocations and maps both
//     logical B generations to one 32 KiB physical allocation.  Both phase
//     groups consume the complete B tile; full-CTA local barriers therefore
//     retire every old-B read before either group starts overwriting B.  The
//     next generation is published only after both groups have contributed
//     their producer slices.
//
//   BM256 x BN128 or BM128 x BN256, BK64 Bit16
//     These have the same 32 KiB/16 KiB physical operand geometry and use the
//     same single-A/single-B ownership and synchronization contracts as the
//     corresponding Bit8 BK128 tiles.
//
// Row-row inputs are accepted only when Pipeline exposes the same unique A/B
// chains, vectorization, physical-buffer ownership, and barrier contract; an
// unmatched spelling is left on the ordinary pipeline.  Autotune must therefore
// compare square, asymmetric, BPP, and non-BPP configurations; enabling BPP
// never asserts that one tile is profitable for every GEMM or scale-GEMM
// shape.
//
// The schedule relies on the following HCU issue rules:
//
//  * An 8-wave workgroup places two waves on each SIMD.  One wave executes an
//    MMAC phase while the other executes a memory/DS phase, then they swap.
//  * MMAC and VALU cannot issue together.  Whichever wave has higher priority
//    can monopolize that shared issue opportunity; equal priority does not
//    guarantee useful alternation, and physical wave0 has an additional small
//    bias.
//  * MMAC can make progress while the peer wave issues SALU, buffer_load, and
//    DS operations.  Memory phases should therefore avoid long VALU address or
//    register-copy chains.
//  * All waves run memory phases at priority 1.  A dot cluster is bracketed by
//    sched_barrier 0, lowered to priority 0 after its first MMAC by the
//    existing HCU dot lowering, and restored to 1 at its end.  Inserting
//    priority toggles inside a run of MMACs does not solve starvation: a
//    low-priority wave cannot execute its later setprio until the high-priority
//    peer gives up the issue slot.
//
// After the AMD Pipeline has materialized two loop-carried LDS slots, the
// common schedule splits one BK iteration into two legal BK/2 dot halves:
//
//   all waves, pipeline prologue:
//     buffer_load A/B(K0), A/B(K1) -> loop-carried VGPR
//     DS_WRITE A/B(K0) -> slot0
//     lgkmcnt(0); full barrier                 publish slot0
//     WaveLow waits at the entry CondBarrier
//     WaveHigh enters M0(K0)
//
//   time       WaveHigh                         WaveLow
//   -----------------------------------------------------------------------
//   start      M0(K0)                           entry CondBarrier
//   P0         D0(K0): MMAC half0               M0(K0)
//   P1         M1(K0)                           D0(K0): MMAC half0
//   P2         D1(K0): MMAC half1               M1(K0)
//   P3         M0(K1)                           D1(K0): MMAC half1
//   P4         D0(K1): MMAC half0               M0(K1)
//   P5         M1(K1)                           D0(K1): MMAC half0
//   P6         D1(K1): MMAC half1               M1(K1)
//
// A Bit8 reduction uses a two-phase whole-BK variant when the dot has a
// separate register-only accumulator update before the loop yield.  Splitting
// such a dot across four phases adds rendezvous without splitting the update,
// while leaving INT32 to FP32 accumulation or block-scale multiplication in
// the following memory phase makes that high-priority wave starve its MMAC
// peer.  BK128 also benefits from amortizing the rendezvous over twice as much
// K work:
//
//   M [read full A+B, issue future A+B, store next A+B] ->
//   D [full BK64/BK128 dot, optional register-only accumulator update]
//
// A direct-FP32 accumulator, such as channel-wise FP8, retains the common
// four-phase schedule at BK64 because it has no post-dot update and benefits
// from the finer-grained overlap.
//
// Bit16 and Bit8 deliberately use different phase builders.  Their first-level
// selection uses only input storage width: Bit16 includes FP16/BF16; Bit8
// includes FP8E4M3/FP8E5M2/i8/u8.  Within Bit8, the dot-to-yield reduction
// dataflow selects the split four-phase or whole-BK two-phase builder.  No
// concrete dtype name controls either decision.
//
// Bit16 phase map (FP16/BF16):
//
//   generic B memdesc (commonly row-col):
//     M0 [read A0+B0, store A(k+1), issue A(k+2)] -> D0 [dot0]
//     M1 [read A1+B1, store B(k+1), issue B(k+2)] -> D1 [dot1]
//
//   panel-major ds_read_m B memdesc (eligible row-row):
//     M0 [read A0+B0+B1, store A(k+1), issue A(k+2)] -> D0 [dot0]
//     M1 [read A1,       store B(k+1), issue B(k+2)] -> D1 [dot1]
//
// Four-phase Bit8 map (FP8E4M3/FP8E5M2/i8/u8):
//
//   generic B memdesc (commonly row-col):
//     M0 [read B0+A0, issue A+B(k+2), read B1]          -> D0 [dot0]
//     M1 [read A1, store A+B(k+1)]                      -> D1 [dot1]
//
//   panel-major ds_read_m B memdesc (eligible row-row):
//     M0 [read B0+A0, issue B+A(k+2), read B1,
//         store A(k+1)]                                 -> D0 [dot0]
//     M1 [read A1, store B(k+1)]                        -> D1 [dot1]
//
//   BM128 x BN128 x BK128 double-A/double-B physical ring:
//     p = current generation parity; p^1 = next generation parity
//
//     M0(k) [read B[p].lo+A[p].lo, issue A+B(k+2), read B[p].hi]
//       -- control barrier --> D0(k) [BK64 dot.lo,
//                                    store A(k+1) -> A[p^1]]
//       -- control barrier -->
//     M1(k) [read A[p].hi, store B(k+1) -> B[p^1]]
//       -- full barrier: publish the complete p^1 generation -->
//     D1(k) [BK64 dot.hi]
//       -- control barrier --> M0(k+1)
//
//     time       WaveHigh                         WaveLow
//     -----------------------------------------------------------------------
//     P0         D0(k)                            M0(k)
//     P1         M1(k)                            D0(k)
//     P2         D1(k)                            M1(k)
//     P3         M0(k+1)                          D1(k)
//
//     storeA may move into D0 only because dot.lo has consumed A[p].lo and
//     A[p^1] is a disjoint allocation; the full barrier remains after storeB
//     in M1 and collectively publishes both operands.  The D1 control barrier
//     is also a data boundary, not optional overhead.  It
//     prevents WaveHigh's M0(k+1) from reading p^1 while WaveLow is still in
//     M1(k) writing its slices of that same generation.  Two physical slots
//     prevent old/new aliasing but do not publish a collectively written tile.
//
// Four-phase Bit8 BM256 x BN128 x BK128 single-A/double-B map:
//
//   M0(k) [read A.lo+B.lo+A.hi, issue A(k+1)]
//     -- full-CTA local barrier after D0/M0 rendezvous -->
//   D0(k) [issue B(k+1), BK64 dot.lo]
//   M1(k) [read B.hi, store B(k+1) -> B[p^1],
//          store A(k+1) -> A, full barrier]
//   D1(k) [BK64 dot.hi]
//
//   time       WaveHigh                         WaveLow
//   -----------------------------------------------------------------------
//   start      M0(K0)                           entry CondBarrier
//   P0         D0(K0)                           M0(K0)
//   P1         M1(K0)                           D0(K0)
//   P2         D1(K0)                           M1(K0)
//   P3         M0(K1)                           D1(K0)
//
// A has two logical generations but one physical allocation.  Both A halves
// are loaded in M0, and the D0/M0 dynamic rendezvous is represented by a
// LocalBarrier.  Therefore every wave has retired every old-A LDS read before
// any wave enters M1 and overwrites A.  B[p] and B[p^1] are distinct physical
// allocations.  No producer/consumer row-ownership coincidence, and no
// LEGACY-versus-INTERLEAVE MMAC layout distinction, is part of this safety
// proof.
//
// Four-phase Bit8 BM128 x BN256 x BK128 double-A/single-B map:
//
//   M0(k) [read B.lo+A.lo+B.hi, issue B(k+1)]
//     -- control barrier after D0/M0 rendezvous -->
//   D0(k) [issue A(k+1), BK64 dot.lo]
//   M1(k) [store A(k+1) -> A[p^1], WaveLow stores its B(k+1) slices,
//          read A.hi, full barrier]
//   D1(k) [BK64 dot.hi, WaveHigh stores its B(k+1) slices,
//          full-CTA local barrier]
//
// Single-B shares the same read-before-overwrite contract but reaches it at a
// later dynamic boundary.  Its store is split between WaveLow M1 and WaveHigh
// D1; neither group can reach a B store before crossing D0's LocalBarrier.
// Thus M0 keeps a control-only rendezvous, while D0 retires every old-B read
// before P2 can overwrite B.  The D1/M1 LocalBarrier then publishes the
// complete collectively written B generation before the next M0 may read it.
//
// The four-phase Bit8 generic and panel variants remain inside one Bit8 path;
// only the panel-sensitive placement of loads/stores differs.  Reading B1
// before D0 covers its LDS latency with dot0.  Generic B normally stores A+B
// together in M1; the measured 128x128x128 double-ring candidate moves only
// storeA after dot0, while panel B overlaps the A1 read with the later B store.
//
// M0 normally ends with a control-only rendezvous.  Single-A is the exception:
// its following peer M1 overwrites A immediately, so M0 must use LocalBarrier.
// Single-B delays every overwrite until both groups have crossed D0's
// LocalBarrier and therefore keeps M0 control-only.  M1 completes A+B(k+1)
// and uses the full gpu.barrier to publish the next LDS generation.  Other
// phase boundaries stay control-only unless they protect a physical-single
// overwrite; unnecessarily representing them as gpu.barrier would make
// MembarAnalysis drain unrelated VMEM/DS work.
//
// Late unroll=2 copies the selected two- or four-phase body.  It means
// 2*BLOCK_SIZE_K (step=64 for BK=32), never a shape-specific 1024 constant.
// The generic LoopUnroll pass remains responsible for a one-BK remainder.

constexpr StringLiteral kCandidateAttr =
    "hcu.block_pingpong.buffer_load_stage3";
constexpr StringLiteral kAppliedAttr = "hcu.block_pingpong.applied";
constexpr StringLiteral kStaticBuffersAttr =
    "hcu.block_pingpong.static_buffers";
constexpr StringLiteral kPingpongBufferAttr = "hcu.block_pingpong.lds_buffer";
constexpr StringLiteral kSingleBufferAttr = "hcu.block_pingpong.single_buffer";
constexpr StringLiteral kMaskedTailAttr = "hcu.block_pingpong.masked_tail";
constexpr StringLiteral kMaskedTailSlicedAttr =
    "hcu.block_pingpong.masked_tail_sliced";

constexpr StringLiteral kLocalStoreStageAttr =
    "ttg.pipeline.hcu_local_store_stage";
constexpr StringLiteral kNumBuffersAttr = "ttg.pipeline.hcu_num_buffers";
constexpr StringLiteral kLoopUnrollFactorAttr = "tt.loop_unroll_factor";

constexpr int32_t kRequiredStages = 3;
constexpr int32_t kRequiredWarps = 8;
constexpr int32_t kRequiredWaveSize = 64;
constexpr int32_t kWaveIdShift = 6;
constexpr int32_t kWaveGroupSize = kRequiredWarps / 2;
constexpr unsigned kNumOperands = 2;
constexpr unsigned kNumKHalves = 2;
constexpr unsigned kOperandA = 0;
constexpr unsigned kOperandB = 1;
constexpr unsigned kFirstKHalf = 0;
constexpr unsigned kSecondKHalf = 1;
constexpr int32_t kNumLDSBuffers = 2;
constexpr int32_t kLocalStoreStage = 1;
constexpr int32_t kLateUnrollFactor = kNumKHalves;
constexpr int64_t kMinimumBlockMN = 128;
constexpr int64_t kBlockMNAlignment = 128;
constexpr int64_t kMaximumAccumulatorBlockMN = 256;
constexpr int64_t kMaximumAccumulatorElements =
    kMaximumAccumulatorBlockMN * kMaximumAccumulatorBlockMN;
constexpr int64_t kRequiredBit8BlockK = 64;
constexpr int64_t kExtendedBit8BlockK = 128;
constexpr int64_t kAsymmetricBit16BlockK = 64;
constexpr int64_t kMaximumLDSBytes = 64 * 1024;
constexpr unsigned kBitsPerByte = 8;
constexpr unsigned kBit8Width = 8;
constexpr unsigned kBit16Width = 16;
constexpr unsigned kRequiredGlobalLoadBits = 128;
constexpr unsigned kPackedWordWidth = 32;
constexpr unsigned kBit8ElementsPerWord = kPackedWordWidth / kBit8Width;
constexpr unsigned kSharedMemoryAddressSpace =
    static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared);
constexpr StringLiteral kPackBlockPingpongBit8Attr =
    "hcu.block_pingpong.pack_bit8_carriers";

static_assert(kRequiredWarps == 2 * kWaveGroupSize);
static_assert(kRequiredWaveSize == (1 << kWaveIdShift));
static_assert(kNumKHalves == kNumLDSBuffers &&
              kNumKHalves == kLateUnrollFactor);

bool hasShape(Type type, ArrayRef<int64_t> expected) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getShape() == expected;
}

bool hasElementBitWidth(Type type, unsigned bitWidth) {
  return type.isIntOrFloat() && type.getIntOrFloatBitWidth() == bitWidth;
}

bool hasSupportedElementType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return false;
  Type elementType = tensorType.getElementType();
  return hasElementBitWidth(elementType, kBit8Width) ||
         hasElementBitWidth(elementType, kBit16Width);
}

unsigned getStorageBitWidth(RankedTensorType type) {
  return type.getElementType().getIntOrFloatBitWidth();
}

bool isDirectChildOf(Operation *op, Operation *parent) {
  return op->getParentOp() == parent;
}

bool hasRequiredWaveConfiguration(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  return module && ttg::lookupNumWarps(op) == kRequiredWarps &&
         ttg::TritonGPUDialect::getThreadsPerWarp(module) == kRequiredWaveSize;
}

struct OriginalCandidate {
  scf::ForOp loop;
  tt::DotOp dot;
  SmallVector<tt::LoadOp, kNumOperands> loads;
  int64_t blockM;
  int64_t blockN;
  int64_t blockK;
};

struct BlockTile {
  int64_t m;
  int64_t n;
  int64_t k;
};

// The asymmetric single-buffer schedules operate on the same 32 KiB/16 KiB
// operand geometry for Bit8 BK128 and Bit16 BK64.  Keep the element width in
// this contract so no other K shape can inherit the overwrite barriers merely
// because its M/N tile matches.
bool hasAsymmetricSingleBufferK(int64_t blockK, unsigned storageBitWidth) {
  return (storageBitWidth == kBit8Width && blockK == kExtendedBit8BlockK) ||
         (storageBitWidth == kBit16Width && blockK == kAsymmetricBit16BlockK);
}

// The 256x128 path aliases the larger A operand and keeps two B generations.
bool usesSingleABuffer(int64_t blockM, int64_t blockN, int64_t blockK,
                       unsigned storageBitWidth) {
  return blockM == 256 && blockN == 128 &&
         hasAsymmetricSingleBufferK(blockK, storageBitWidth);
}

// The transposed 128x256 path keeps A as an ordinary two-slot ring and aliases
// B's two logical generations. Both asymmetric paths use a full-CTA local
// barrier to retire every read of their complete old aliased operand before
// either group starts overwriting it.
bool usesSingleBBuffer(int64_t blockM, int64_t blockN, int64_t blockK,
                       unsigned storageBitWidth) {
  return blockM == 128 && blockN == 256 &&
         hasAsymmetricSingleBufferK(blockK, storageBitWidth);
}

// Find the unique source tt.load feeding one dot operand.  Walking the actual
// def-use chain avoids guessing A/B from the number or order of loads in the K
// loop, and remains unambiguous when two logical operand tiles have the same
// shape.  More than one load means the operand is assembled or dequantized in
// the loop; slicing such a value requires a different semantic contract.
FailureOr<tt::LoadOp> findUniqueOperandLoad(Value operand, scf::ForOp loop) {
  Operation *root = operand.getDefiningOp();
  if (!root || root->getBlock() != loop.getBody())
    return failure();

  SetVector<Operation *> dependencies;
  dependencies.insert(root);
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  options.filter = [&](Operation *dependency) {
    return dependency->getBlock() == loop.getBody();
  };
  (void)getBackwardSlice(root, &dependencies, options);

  tt::LoadOp sourceLoad;
  for (Operation *dependency : dependencies) {
    auto load = dyn_cast<tt::LoadOp>(dependency);
    if (!load)
      continue;
    if (sourceLoad)
      return failure();
    sourceLoad = load;
  }
  if (!sourceLoad)
    return failure();
  return sourceLoad;
}

// Phase scheduling moves the A/B load/store chains within the loop body.  A
// read-only auxiliary load is safe, but crossing an unmodelled write, atomic,
// barrier, or other side effect is not.  Volatile reads are also ordering
// sensitive and are outside this schedule family.
bool hasUnsupportedSourceSideEffects(scf::ForOp loop) {
  return llvm::any_of(loop.getBody()->without_terminator(), [&](Operation &op) {
    if (auto load = dyn_cast<tt::LoadOp>(op))
      return load.getIsVolatile();
    // AccelerateMatmul may already express a row-row dot operand as
    // tt.load -> local_alloc -> local_load before Prepare runs.  This is a
    // private conversion buffer, not an independently ordered side effect;
    // AMD Pipeline later folds it into the same operand chain rematched by
    // BlockPingpong.
    if (isa<ttg::LocalAllocOp, ttg::LocalLoadOp>(op))
      return false;
    return !isMemoryEffectFree(&op);
  });
}

// Recover the logical GEMM tile [BM, BK] x [BK, BN] -> [BM, BN] from the dot
// and return {BM, BN, BK}.  Besides basic rank/shape consistency, this helper
// validates the measured M/N workload, MFMA-compatible BK/2 halves, operand
// types/encodings, and the 64 KiB budget for the selected physical LDS scheme.
FailureOr<BlockTile> getSupportedTile(tt::DotOp dot) {
  auto resultType = dyn_cast<RankedTensorType>(dot.getType());
  auto aType = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto bType = dyn_cast<RankedTensorType>(dot.getB().getType());
  if (!resultType || !aType || !bType || resultType.getRank() != 2 ||
      aType.getRank() != 2 || bType.getRank() != 2)
    return failure();

  int64_t blockM = resultType.getShape()[0];
  int64_t blockN = resultType.getShape()[1];
  int64_t blockK = aType.getShape()[1];
  if (aType.getShape()[0] != blockM || bType.getShape()[0] != blockK ||
      bType.getShape()[1] != blockN)
    return failure();

  // The two wave groups need enough independent MMAC work in every phase.
  // Keep M/N on the measured 128-element granularity and cap the accumulator
  // footprint at the 256x256 case, which leaves headroom under HCU's 256-VGPR
  // limit for the loop-carried A/B prefetch values.
  if (blockM < kMinimumBlockMN || blockN < kMinimumBlockMN ||
      blockM % kBlockMNAlignment != 0 || blockN % kBlockMNAlignment != 0 ||
      blockM > kMaximumAccumulatorElements / blockN)
    return failure();

  if (!hasSupportedElementType(aType) || !hasSupportedElementType(bType) ||
      getStorageBitWidth(aType) != getStorageBitWidth(bType))
    return failure();

  // HCU's native 8-bit MMAC consumes K=32 per instruction.  BK64 uses the
  // normal two-generation A+B ring. BK128 uses that ring for 128x128, and the
  // proved single-A/single-B contract for the two asymmetric shapes. Each
  // BK128 dot contains four native MMAC K steps, with phase selection handled
  // below.
  bool isBit8 = hasElementBitWidth(aType.getElementType(), kBit8Width);
  if (isBit8 && blockK != kRequiredBit8BlockK && blockK != kExtendedBit8BlockK)
    return failure();
  if (isBit8 && blockK == kExtendedBit8BlockK) {
    // The measured BK128 family carries the dot result directly.  Do not also
    // admit a post-dot reduction shape at this larger K until its live ranges
    // and whole-tile schedule have independent coverage.
    auto loop = dot->getParentOfType<scf::ForOp>();
    auto yield = loop ? dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator())
                      : nullptr;
    if (!yield || !llvm::is_contained(yield.getOperands(), dot.getResult()))
      return failure();
  }

  auto aEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(aType.getEncoding());
  auto bEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(bType.getEncoding());
  auto mfmaEncoding =
      aEncoding ? dyn_cast<ttg::AMDMfmaEncodingAttr>(aEncoding.getParent())
                : nullptr;
  if (!aEncoding || !bEncoding || !mfmaEncoding ||
      aEncoding.getParent() != bEncoding.getParent() ||
      aEncoding.getKWidth() != bEncoding.getKWidth() ||
      blockK % kNumKHalves != 0)
    return failure();

  // M0/D0 and M1/D1 each consume BK/2.  Both halves must contain complete
  // MFMA K instructions; otherwise slicing the original dot would create an
  // operand shape that the HCU dot lowering cannot represent.
  auto aInstrShape =
      mfmaEncoding.getInstrShapeForOperand(aEncoding.getKWidth(), kOperandA);
  auto bInstrShape =
      mfmaEncoding.getInstrShapeForOperand(bEncoding.getKWidth(), kOperandB);
  int64_t halfK = blockK / kNumKHalves;
  if (aInstrShape.size() != 2 || bInstrShape.size() != 2 ||
      halfK < aInstrShape[1] || halfK < bInstrShape[0] ||
      halfK % aInstrShape[1] != 0 || halfK % bInstrShape[0] != 0)
    return failure();

  // The proved asymmetric BK128 extensions make the larger operand physical
  // single-buffered.  Both 256x128 single-A and 128x256 single-B use the same
  // full-CTA contract: all waves finish reading the old aliased generation
  // before any wave overwrites it.  Other families retain two complete A+B
  // generations.
  unsigned storageBitWidth = getStorageBitWidth(aType);
  int64_t aTileBits = blockM * blockK * storageBitWidth;
  int64_t bTileBits = blockK * blockN * storageBitWidth;
  int64_t requiredLDSBits = kNumLDSBuffers * (aTileBits + bTileBits);
  if (usesSingleABuffer(blockM, blockN, blockK, storageBitWidth))
    requiredLDSBits = aTileBits + kNumLDSBuffers * bTileBits;
  else if (usesSingleBBuffer(blockM, blockN, blockK, storageBitWidth))
    requiredLDSBits = kNumLDSBuffers * aTileBits + bTileBits;
  if (requiredLDSBits > kMaximumLDSBytes * kBitsPerByte)
    return failure();

  return BlockTile{blockM, blockN, blockK};
}

// Match the source GEMM loop before AMD Pipeline expands global loads through
// LDS.  This stage accepts one direct-child dot with one ordinary BufferLoad
// source per operand. Independent auxiliary reads are allowed; all shape,
// type, MFMA, LDS, side-effect, and control-flow checks are completed before
// Prepare attaches the cross-pass candidate marker.
FailureOr<OriginalCandidate> matchOriginalCandidate(scf::ForOp loop) {
  if (!hasRequiredWaveConfiguration(loop))
    return failure();

  SmallVector<tt::DotOp> dots;
  SmallVector<tt::DotScaledOp> scaledDots;
  SmallVector<tt::LoadOp> loads;
  bool hasUnsupportedCopy = false;
  bool hasNestedControl = false;

  loop.walk([&](Operation *op) {
    if (isa<tt::MatrixLoadOp, tta::MatrixLoadToLocalOp,
            ttg::AsyncCopyGlobalToLocalOp>(op))
      hasUnsupportedCopy = true;
    if (auto dot = dyn_cast<tt::DotOp>(op))
      dots.push_back(dot);
    if (auto dot = dyn_cast<tt::DotScaledOp>(op))
      scaledDots.push_back(dot);
    if (auto load = dyn_cast<tt::LoadOp>(op))
      loads.push_back(load);
    if (op != loop && op->getNumRegions() != 0)
      hasNestedControl = true;
  });

  if (hasUnsupportedCopy || hasNestedControl || dots.size() != 1 ||
      !scaledDots.empty() || hasUnsupportedSourceSideEffects(loop))
    return failure();

  tt::DotOp dot = dots.front();
  // The phase scheduler moves these operations within one loop-body block.
  // Reject a dot or load nested under scf.if, ttg.mask, or another region so
  // the transformation never crosses a control-flow boundary.
  if (!isDirectChildOf(dot, loop) || !llvm::all_of(loads, [&](tt::LoadOp load) {
        return isDirectChildOf(load, loop);
      }))
    return failure();

  auto loadA = findUniqueOperandLoad(dot.getA(), loop);
  auto loadB = findUniqueOperandLoad(dot.getB(), loop);
  if (failed(loadA) || failed(loadB) || *loadA == *loadB ||
      !isDirectChildOf(*loadA, loop) || !isDirectChildOf(*loadB, loop))
    return failure();

  auto supportedTile = getSupportedTile(dot);
  if (failed(supportedTile))
    return failure();

  auto [blockM, blockN, blockK] = *supportedTile;

  if (!hasShape(loadA->getType(), {blockM, blockK}) ||
      !hasShape(loadB->getType(), {blockK, blockN}))
    return failure();

  SmallVector<tt::LoadOp, kNumOperands> operandLoads{*loadA, *loadB};
  return OriginalCandidate{loop,   dot,    std::move(operandLoads),
                           blockM, blockN, blockK};
}

// Require both A and B to retain at least one 128-bit global-load vector.
// BlockPingpong keeps prefetched tiles live in VGPRs across an extra pipeline
// stage; narrow/scalar loads multiply the instruction count and can push this
// live range into spilling.  A mask may reduce the legal width, so use the
// smaller of pointer contiguity and mask alignment for the final check.
bool hasRequiredGlobalLoadVectorization(
    const OriginalCandidate &candidate,
    triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const DenseMap<Operation *, Value> *effectiveMasks = nullptr) {
  return llvm::all_of(candidate.loads, [&](tt::LoadOp load) {
    auto loadType = cast<RankedTensorType>(load.getType());
    unsigned vectorElements = axisInfoAnalysis.getContiguity(load.getPtr());
    Value mask = load.getMask();
    if (effectiveMasks) {
      auto maskIt = effectiveMasks->find(load);
      if (maskIt != effectiveMasks->end())
        mask = maskIt->second;
    }
    if (mask)
      vectorElements =
          std::min(vectorElements, axisInfoAnalysis.getMaskAlignment(mask));
    unsigned vectorBits = vectorElements * loadType.getElementTypeBitWidth();
    if (vectorBits < kRequiredGlobalLoadBits) {
      LDBG("Reject HCU BufferLoad BlockPingpong: load vector width is "
           << vectorBits << " bits, required " << kRequiredGlobalLoadBits);
      return false;
    }
    return true;
  });
}

// Python's tl.max_constancy lowers to a tt.constancy hint on the operation
// defining the hinted value.  Preserve the same information here after the
// HCU matcher has proved that a loop-invariant mask supports the required
// 128-bit load.  In particular, a MoE token-validity predicate is constant
// across A's K dimension, but cloning the unmasked main loop can
// otherwise lose that fact before AMD ScheduleLoops filters small loads.
void preserveMaskConstancyHints(
    const OriginalCandidate &candidate,
    triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const DenseMap<Operation *, Value> *effectiveMasks = nullptr) {
  for (tt::LoadOp load : candidate.loads) {
    Value mask = load.getMask();
    if (effectiveMasks) {
      auto maskIt = effectiveMasks->find(load);
      if (maskIt != effectiveMasks->end())
        mask = maskIt->second;
    }
    if (!mask)
      continue;

    Operation *definingOp = mask.getDefiningOp();
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    if (!definingOp || !maskType)
      continue;

    auto *axisInfo = axisInfoAnalysis.getAxisInfo(mask);
    if (!axisInfo)
      continue;

    SmallVector<int32_t> constancy;
    llvm::transform(axisInfo->getConstancy(), std::back_inserter(constancy),
                    [](int64_t value) { return static_cast<int32_t>(value); });
    auto hintType = RankedTensorType::get(
        {maskType.getRank()}, IntegerType::get(mask.getContext(), 32));
    definingOp->setDiscardableAttr(
        "tt.constancy", DenseIntElementsAttr::get(hintType, constancy));
  }
}

// Recognize the canonical K-tail predicate used by the A/B loads:
//
//   broadcast(cmpi slt(add(splat(loop.iv), kOffset), splat(loop.ub)))
//
// A uses a [1, BK] source predicate and B uses [BK, 1].  BlockPingpong strips
// only this loop-variant K predicate from the unmasked main loop; the original
// predicate remains on the one-BK tail.  Keep the matcher deliberately narrow:
// an unfamiliar mask makes the candidate fail instead of risking an invalid
// main/tail split.
bool isKBoundaryMask(Value mask, scf::ForOp loop,
                     ArrayRef<int64_t> expectedMaskSourceShape) {
  auto broadcast = mask ? mask.getDefiningOp<tt::BroadcastOp>() : nullptr;
  if (!broadcast ||
      !hasShape(broadcast.getSrc().getType(), expectedMaskSourceShape))
    return false;

  auto compare = broadcast.getSrc().getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt)
    return false;

  auto upperSplat = compare.getRhs().getDefiningOp<tt::SplatOp>();
  if (!upperSplat || upperSplat.getSrc() != loop.getUpperBound())
    return false;

  auto add = compare.getLhs().getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  auto lhsSplat = add.getLhs().getDefiningOp<tt::SplatOp>();
  auto rhsSplat = add.getRhs().getDefiningOp<tt::SplatOp>();
  bool lhsIsIV = lhsSplat && lhsSplat.getSrc() == loop.getInductionVar();
  bool rhsIsIV = rhsSplat && rhsSplat.getSrc() == loop.getInductionVar();
  if (lhsIsIV == rhsIsIV)
    return false;

  // Only the IV may vary with the loop.  Requiring the other operand to be a
  // loop-invariant K offset prevents an unrelated data-dependent predicate
  // from being stripped from the unmasked main loop.
  Value kOffset = lhsIsIV ? add.getRhs() : add.getLhs();
  return loop.isDefinedOutsideOfLoop(kOffset);
}

// Compute the mask that must remain on a load in the unmasked main K loop.
// A null Value is a successful result meaning that the load becomes unmasked;
// failure means the original predicate cannot be decomposed safely.
FailureOr<Value> getMainLoopMask(tt::LoadOp load, scf::ForOp loop,
                                 ArrayRef<int64_t> expectedMaskSourceShape) {
  Value mask = load.getMask();
  if (isKBoundaryMask(mask, loop, expectedMaskSourceShape))
    return Value{};

  // MoE A loads combine a loop-invariant padded-token predicate with the
  // loop-variant K-tail predicate. The unmasked main K range must retain the
  // token predicate while dropping only the K component.
  auto andOp = mask ? mask.getDefiningOp<arith::AndIOp>() : nullptr;
  if (!andOp)
    return failure();
  // `and` is commutative, so accept either operand order.  Preserve only the
  // loop-invariant predicate (for example, the MoE token-validity mask) on the
  // main loop and leave the complete combined predicate on the K-tail loop.
  if (isKBoundaryMask(andOp.getLhs(), loop, expectedMaskSourceShape) &&
      loop.isDefinedOutsideOfLoop(andOp.getRhs()))
    return andOp.getRhs();
  if (isKBoundaryMask(andOp.getRhs(), loop, expectedMaskSourceShape) &&
      loop.isDefinedOutsideOfLoop(andOp.getLhs()))
    return andOp.getLhs();
  return failure();
}

// Clone a loop with new bounds/init arguments while replacing selected load
// masks for the aligned main K range.  The explicit DCE keeps removed K-mask
// calculation chains from reaching AMD's scheduler.
scf::ForOp cloneLoop(OpBuilder &builder, scf::ForOp source, Value lowerBound,
                     Value upperBound, ValueRange initArgs,
                     const DenseMap<Operation *, Value> &mainLoopMasks,
                     IRMapping &mapping) {
  auto clone = scf::ForOp::create(builder, source.getLoc(), lowerBound,
                                  upperBound, source.getStep(), initArgs);
  clone->setAttrs(source->getAttrs());
  if (clone.getBody()->mightHaveTerminator())
    clone.getBody()->getTerminator()->erase();

  mapping.map(source.getInductionVar(), clone.getInductionVar());
  for (auto [oldArg, newArg] :
       llvm::zip(source.getRegionIterArgs(), clone.getRegionIterArgs()))
    mapping.map(oldArg, newArg);

  builder.setInsertionPointToEnd(clone.getBody());
  for (Operation &op : source.getBody()->getOperations()) {
    auto load = dyn_cast<tt::LoadOp>(op);
    auto maskIt = load ? mainLoopMasks.find(load) : mainLoopMasks.end();
    if (maskIt != mainLoopMasks.end()) {
      Value ptr = mapping.lookupOrDefault(load.getPtr());
      tt::LoadOp replacement;
      if (Value mainMask = maskIt->second) {
        mainMask = mapping.lookupOrDefault(mainMask);
        if (Value other = load.getOther())
          replacement = tt::LoadOp::create(
              builder, load.getLoc(), ptr, mainMask,
              mapping.lookupOrDefault(other), load.getCache(), load.getEvict(),
              load.getIsVolatile());
        else
          replacement = tt::LoadOp::create(
              builder, load.getLoc(), ptr, mainMask, load.getCache(),
              load.getEvict(), load.getIsVolatile());
      } else {
        replacement =
            tt::LoadOp::create(builder, load.getLoc(), ptr, load.getCache(),
                               load.getEvict(), load.getIsVolatile());
      }
      mapping.map(load.getResult(), replacement.getResult());
      continue;
    }
    builder.clone(op, mapping);
  }

  // Removing the load masks leaves their IV-dependent comparison chains dead
  // in the main loop.  Delete only trivially dead operations here so AMD's
  // scheduler sees the same compact body as a naturally unmasked kernel.
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> operations;
    for (Operation &op : clone.getBody()->without_terminator())
      operations.push_back(&op);
    for (Operation *op : llvm::reverse(operations)) {
      if (!isOpTriviallyDead(op))
        continue;
      op->erase();
      changed = true;
    }
  } while (changed);
  return clone;
}

struct KBoundaryTailPlan {
  OriginalCandidate candidate;
  tt::LoadOp loadA;
  tt::LoadOp loadB;
  Value aMainMask;
  Value bMainMask;
  unsigned accumulatorIndex;
  scf::YieldOp oldYield;
  SmallVector<Value> pointerOffsets;
};

// Validate every requirement of the masked-main/tail rewrite without creating
// IR.  Keeping analysis separate from mutation guarantees that a rejected
// candidate leaves the original loop byte-for-byte unchanged.
FailureOr<KBoundaryTailPlan>
buildKBoundaryTailPlan(OriginalCandidate candidate) {
  scf::ForOp loop = candidate.loop;
  APInt lowerBound;
  APInt step;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lowerBound)) ||
      lowerBound.getSExtValue() != 0 ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
      step.getSExtValue() != candidate.blockK)
    return failure();

  tt::LoadOp loadA = candidate.loads[kOperandA];
  tt::LoadOp loadB = candidate.loads[kOperandB];
  if (!loadA || !loadB || !loadA.getMask() || !loadB.getMask())
    return failure();
  auto aMainMask = getMainLoopMask(loadA, loop, {1, candidate.blockK});
  auto bMainMask = getMainLoopMask(loadB, loop, {candidate.blockK, 1});
  if (failed(aMainMask) || failed(bMainMask))
    return failure();

  std::optional<unsigned> accumulatorIndex;
  for (auto [index, result] : llvm::enumerate(loop.getResults())) {
    if (result.getType() != candidate.dot.getType())
      continue;
    if (accumulatorIndex)
      return failure();
    accumulatorIndex = index;
  }
  if (!accumulatorIndex)
    return failure();
  for (auto [index, result] : llvm::enumerate(loop.getResults())) {
    if (index != *accumulatorIndex && !result.use_empty())
      return failure();
  }

  auto oldYield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<Value> pointerOffsets(loop.getInitArgs().size());
  for (auto [index, blockArg] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (index == *accumulatorIndex)
      continue;
    auto increment = oldYield.getOperand(index).getDefiningOp<tt::AddPtrOp>();
    if (!increment || increment.getPtr() != blockArg)
      return failure();
    Value offset = increment.getOffset();
    if (Operation *offsetDef = offset.getDefiningOp();
        offsetDef && loop->isAncestor(offsetDef))
      return failure();
    pointerOffsets[index] = offset;
  }

  return KBoundaryTailPlan{candidate,  loadA,
                           loadB,      *aMainMask,
                           *bMainMask, *accumulatorIndex,
                           oldYield,   std::move(pointerOffsets)};
}

DenseMap<Operation *, Value> getMainLoopMasks(const KBoundaryTailPlan &plan) {
  return {{plan.loadA, plan.aMainMask}, {plan.loadB, plan.bMainMask}};
}

// Apply a fully validated plan.  Pointer loop-carried values are advanced to
// mainUpper explicitly, while the accumulator flows main -> tail -> users.
// No fallible matching remains after mutation starts.
OriginalCandidate applyKBoundaryTailPlan(const KBoundaryTailPlan &plan) {
  OriginalCandidate candidate = plan.candidate;
  scf::ForOp loop = candidate.loop;
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  Value span = arith::SubIOp::create(builder, loc, loop.getUpperBound(),
                                     loop.getLowerBound());
  Value remainder = arith::RemUIOp::create(builder, loc, span, loop.getStep());
  Value mainUpper =
      arith::SubIOp::create(builder, loc, loop.getUpperBound(), remainder);
  // mainUpper = upper - (upper - lower) % BK is aligned to BK.  Replacing the
  // original step-aligned loop IV with this scalar value otherwise loses that
  // fact in AxisInfo, reducing the one-shot tail mask alignment to one fp16
  // element and scalarizing an otherwise legal buffer_load_dwordx4.  Preserve
  // the same divisibility that the original loop IV obtained from step=BK;
  // the load lowering still intersects it with the runtime K divisibility and
  // therefore only vectorizes widths proven safe for the concrete signature.
  mainUpper.getDefiningOp()->setAttr(
      "tt.divisibility", builder.getI32IntegerAttr(candidate.blockK));

  Value numMainIterations =
      arith::DivUIOp::create(builder, loc, mainUpper, loop.getStep());
  SmallVector<Value> tailInitArgs(loop.getInitArgs());
  for (auto [index, offset] : llvm::enumerate(plan.pointerOffsets)) {
    if (index == plan.accumulatorIndex)
      continue;
    Value iterationScale = numMainIterations;
    if (isa<RankedTensorType>(offset.getType()))
      iterationScale = tt::SplatOp::create(builder, loc, offset.getType(),
                                           numMainIterations);
    Value scaledOffset =
        arith::MulIOp::create(builder, loc, offset, iterationScale);
    tailInitArgs[index] =
        tt::AddPtrOp::create(builder, loc, loop.getInitArgs()[index].getType(),
                             loop.getInitArgs()[index], scaledOffset);
  }

  Value zero = arith::ConstantOp::create(
      builder, loc, builder.getIntegerAttr(remainder.getType(), 0));
  Value hasTail = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                        remainder, zero);

  DenseMap<Operation *, Value> mainLoopMasks = getMainLoopMasks(plan);
  IRMapping mainMapping;
  scf::ForOp main = cloneLoop(builder, loop, loop.getLowerBound(), mainUpper,
                              loop.getInitArgs(), mainLoopMasks, mainMapping);
  auto mainDot = mainMapping.lookupOrDefault(candidate.dot.getResult())
                     .getDefiningOp<tt::DotOp>();
  SmallVector<tt::LoadOp, kNumOperands> mainLoads;
  for (tt::LoadOp load : candidate.loads) {
    auto mainLoad = mainMapping.lookupOrDefault(load.getResult())
                        .getDefiningOp<tt::LoadOp>();
    assert(mainLoad && "validated main-loop load must be cloned");
    mainLoads.push_back(mainLoad);
  }
  assert(mainDot && "validated main-loop dot must be cloned");
  OriginalCandidate mainCandidate{main,
                                  mainDot,
                                  std::move(mainLoads),
                                  candidate.blockM,
                                  candidate.blockN,
                                  candidate.blockK};
  // Keep the conventional K order: execute the unmasked pipelined main range
  // first, then accumulate the one-shot masked remainder.  The tail remains
  // HCU-local and is never scheduled as BlockPingpong.

  builder.setInsertionPointAfter(main);
  auto tail = scf::IfOp::create(
      builder, loc, TypeRange{loop.getResult(plan.accumulatorIndex).getType()},
      hasTail, true);
  tail->setAttr(kMaskedTailAttr, builder.getUnitAttr());

  IRMapping mapping;
  mapping.map(loop.getInductionVar(), mainUpper);
  for (auto [index, oldArg] : llvm::enumerate(loop.getRegionIterArgs()))
    mapping.map(oldArg, index == plan.accumulatorIndex
                            ? main.getResult(plan.accumulatorIndex)
                            : tailInitArgs[index]);
  OpBuilder thenBuilder = tail.getThenBodyBuilder();
  for (Operation &op : loop.getBody()->without_terminator())
    thenBuilder.clone(op, mapping);
  // The then branch executes the original body once at `mainUpper`; return its
  // mapped accumulator yield.  The else branch has no K-tail and forwards the
  // accumulator produced by the unmasked main loop unchanged.
  scf::YieldOp::create(thenBuilder, loc,
                       mapping.lookupOrDefault(
                           plan.oldYield->getOperand(plan.accumulatorIndex)));
  OpBuilder elseBuilder = tail.getElseBodyBuilder();
  scf::YieldOp::create(elseBuilder, loc, main.getResult(plan.accumulatorIndex));

  loop.getResult(plan.accumulatorIndex).replaceAllUsesWith(tail.getResult(0));
  loop.erase();
  return mainCandidate;
}

// Split the one-shot masked tail dot into two ordered BK/2 dots.  The tail does
// not participate in phase pingpong, but halving its A/B local-load live ranges
// prevents the full-BK operands from exceeding HCU's 256-VGPR limit.
LogicalResult sliceMaskedTailDot(Operation *tail) {
  if (!tail->hasAttr(kMaskedTailAttr) || tail->hasAttr(kMaskedTailSlicedAttr))
    return failure();

  SmallVector<tt::DotOp> dots;
  tail->walk([&](tt::DotOp dot) { dots.push_back(dot); });
  if (dots.size() != 1)
    return failure();
  tt::DotOp dot = dots.front();
  auto supportedTile = getSupportedTile(dot);
  if (failed(supportedTile))
    return failure();
  auto [blockM, blockN, blockK] = *supportedTile;
  auto localA = dot.getA().getDefiningOp<ttg::LocalLoadOp>();
  auto localB = dot.getB().getDefiningOp<ttg::LocalLoadOp>();
  if (!localA || !localB || !localA->hasOneUse() || !localB->hasOneUse() ||
      !hasShape(dot.getA().getType(), {blockM, blockK}) ||
      !hasShape(dot.getB().getType(), {blockK, blockN}))
    return failure();

  auto aType = cast<RankedTensorType>(dot.getA().getType());
  auto bType = cast<RankedTensorType>(dot.getB().getType());
  auto aEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(aType.getEncoding());
  auto bEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(bType.getEncoding());
  auto aMemDesc = dyn_cast<ttg::MemDescType>(localA.getSrc().getType());
  auto bMemDesc = dyn_cast<ttg::MemDescType>(localB.getSrc().getType());
  if (!aEncoding || !bEncoding || !aMemDesc || !bMemDesc ||
      aEncoding.getKWidth() != bEncoding.getKWidth())
    return failure();

  OpBuilder builder(dot);
  Attribute dotEncoding = dot.getType().getEncoding();
  auto createHalfLoad = [&](ttg::LocalLoadOp fullLoad,
                            ttg::MemDescType memDescType, unsigned operandIndex,
                            unsigned half) -> Value {
    unsigned kDimension = operandIndex == 0 ? 1 : 0;
    SmallVector<int64_t> shape(memDescType.getShape());
    shape[kDimension] = blockK / kNumKHalves;
    auto sliceType = ttg::MemDescType::get(
        shape, memDescType.getElementType(), memDescType.getEncoding(),
        memDescType.getMemorySpace(), memDescType.getMutableMemory(),
        memDescType.getAllocShape());
    SmallVector<int32_t> offsets(2, 0);
    offsets[kDimension] = half * (blockK / kNumKHalves);
    auto subview = ttg::MemDescSubsliceOp::create(
        builder, fullLoad.getLoc(), sliceType, fullLoad.getSrc(), offsets);
    auto halfEncoding = ttg::DotOperandEncodingAttr::get(
        builder.getContext(), operandIndex, dotEncoding, aEncoding.getKWidth());
    auto halfType = RankedTensorType::get(shape, memDescType.getElementType(),
                                          halfEncoding);
    return ttg::LocalLoadOp::create(builder, fullLoad.getLoc(), halfType,
                                    subview.getResult());
  };

  Operation *previousDot = nullptr;
  for (unsigned half = 0; half < kNumKHalves; ++half) {
    Value halfA = createHalfLoad(localA, aMemDesc, 0, half);
    Value halfB = createHalfLoad(localB, bMemDesc, 1, half);
    IRMapping mapping;
    mapping.map(dot.getA(), halfA);
    mapping.map(dot.getB(), halfB);
    if (previousDot)
      mapping.map(dot.getC(), previousDot->getResult(0));
    previousDot = builder.clone(*dot, mapping);
  }

  dot->replaceAllUsesWith(previousDot->getResults());
  dot.erase();
  localA.erase();
  localB.erase();
  tail->setAttr(kMaskedTailSlicedAttr, UnitAttr::get(tail->getContext()));
  return success();
}

struct ExpandedCandidate {
  scf::ForOp loop;
  tt::DotOp dot;
  tt::LoadOp globalA;
  tt::LoadOp globalB;
  ttg::LocalLoadOp localA;
  ttg::LocalLoadOp localB;
  ttg::LocalStoreOp storeA;
  ttg::LocalStoreOp storeB;
  ttg::LocalAllocOp allocA;
  ttg::LocalAllocOp allocB;
  int64_t blockM;
  int64_t blockN;
  int64_t blockK;
};

struct StaticBufferPlan {
  ttg::LocalAllocOp combinedAlloc;
  SmallVector<ttg::MemDescIndexOp, kNumLDSBuffers> hotIndices;
};

// Match the canonical two-slot ring update `(index + 1 < 2) ? index + 1 : 0`
// emitted by AMD Pipeline and return the current index feeding the increment.
FailureOr<Value> matchModuloTwoIncrement(Value result) {
  auto select = result.getDefiningOp<arith::SelectOp>();
  if (!select)
    return failure();

  APInt falseValue;
  if (!matchPattern(select.getFalseValue(), m_ConstantInt(&falseValue)) ||
      falseValue.getSExtValue() != 0)
    return failure();

  auto add = select.getTrueValue().getDefiningOp<arith::AddIOp>();
  if (!add)
    return failure();
  APInt increment;
  Value base;
  if (matchPattern(add.getRhs(), m_ConstantInt(&increment)) &&
      increment.getSExtValue() == 1) {
    base = add.getLhs();
  } else if (matchPattern(add.getLhs(), m_ConstantInt(&increment)) &&
             increment.getSExtValue() == 1) {
    base = add.getRhs();
  } else {
    return failure();
  }

  auto compare = select.getCondition().getDefiningOp<arith::CmpIOp>();
  APInt limit;
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt ||
      compare.getLhs() != add.getResult() ||
      !matchPattern(compare.getRhs(), m_ConstantInt(&limit)) ||
      limit.getSExtValue() != kNumLDSBuffers)
    return failure();
  return base;
}

bool requiresStaticDoubleBufferMaterialization(scf::ForOp loop) {
  if (!loop->hasAttr(kAppliedAttr) || loop->hasAttr(kStaticBuffersAttr) ||
      loop->hasAttr(kLoopUnrollFactorAttr))
    return false;
  if (auto stages = loop->getAttrOfType<IntegerAttr>(tt::kNumStagesAttrName))
    return stages.getInt() != 1;
  return true;
}

// After late unroll=2, identify the A and B two-slot allocations and prove that
// their two hot MemDescIndex operations are the consecutive slot1/slot0 pair.
// The proof also verifies the loop-carried ring/index ties before Finalize
// replaces the combined allocations.
FailureOr<SmallVector<StaticBufferPlan, kNumLDSBuffers>>
matchStaticDoubleBuffers(scf::ForOp loop) {
  if (!requiresStaticDoubleBufferMaterialization(loop))
    return failure();

  DenseMap<Operation *, SmallVector<ttg::MemDescIndexOp, kNumLDSBuffers>>
      groupedIndices;
  bool hasMarkedAllocation = false;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto index = dyn_cast<ttg::MemDescIndexOp>(op);
    if (!index)
      continue;
    auto alloc = index.getSrc().getDefiningOp<ttg::LocalAllocOp>();
    if (alloc) {
      groupedIndices[alloc].push_back(index);
      hasMarkedAllocation |= alloc->hasAttr(kPingpongBufferAttr);
    }
  }
  // Normal pipelines carry explicit marks from the successfully scheduled
  // A/B chains.  Retain the unmarked fallback for standalone post-unroll IR
  // accepted by this finalization pass.
  if (hasMarkedAllocation) {
    SmallVector<Operation *> unmarkedAllocations;
    for (auto &entry : groupedIndices)
      if (!entry.first->hasAttr(kPingpongBufferAttr))
        unmarkedAllocations.push_back(entry.first);
    for (Operation *alloc : unmarkedAllocations)
      groupedIndices.erase(alloc);
  }
  if (groupedIndices.size() != kNumOperands)
    return failure();

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<StaticBufferPlan, kNumLDSBuffers> plans;
  for (auto &[allocOp, hotIndices] : groupedIndices) {
    auto alloc = cast<ttg::LocalAllocOp>(allocOp);
    auto allocType = alloc.getType();
    if (allocType.getShape().empty() ||
        allocType.getShape().front() != kNumLDSBuffers ||
        hotIndices.size() != kLateUnrollFactor ||
        hotIndices[0].getType() != hotIndices[1].getType())
      return failure();

    auto ringIndex = matchModuloTwoIncrement(hotIndices[0].getIndex());
    auto secondBase = matchModuloTwoIncrement(hotIndices[1].getIndex());
    if (failed(ringIndex) || failed(secondBase) ||
        *secondBase != hotIndices[0].getIndex())
      return failure();
    auto ringArgument = dyn_cast<BlockArgument>(*ringIndex);
    if (!ringArgument || ringArgument.getOwner() != loop.getBody() ||
        ringArgument.getArgNumber() == 0)
      return failure();
    unsigned ringIterIndex = ringArgument.getArgNumber() - 1;
    APInt initialRingIndex;
    if (ringIterIndex >= loop.getInitArgs().size() ||
        !matchPattern(loop.getInitArgs()[ringIterIndex],
                      m_ConstantInt(&initialRingIndex)) ||
        initialRingIndex.getSExtValue() != 0 ||
        yield.getOperand(ringIterIndex) != hotIndices[1].getIndex())
      return failure();

    std::optional<unsigned> carriedIndex;
    for (auto [index, init] : llvm::enumerate(loop.getInitArgs())) {
      auto initView = init.getDefiningOp<ttg::MemDescIndexOp>();
      if (!initView || initView.getSrc() != alloc.getResult())
        continue;
      APInt initialSlot;
      if (!matchPattern(initView.getIndex(), m_ConstantInt(&initialSlot)) ||
          initialSlot.getSExtValue() != 0 || carriedIndex)
        return failure();
      carriedIndex = index;
    }
    if (!carriedIndex ||
        yield.getOperand(*carriedIndex) != hotIndices.back().getResult())
      return failure();

    for (Operation *user : alloc.getResult().getUsers()) {
      if (!isa<ttg::MemDescIndexOp, ttg::LocalDeallocOp>(user))
        return failure();
    }
    plans.push_back({alloc, std::move(hotIndices)});
  }
  return plans;
}

// Replace a view of combined [2, ...] storage with one of two independent
// allocations.  Hot constant indices fold directly; cold epilogue indices use
// an scf.if because their runtime parity is no longer statically known.
Value selectStaticBuffer(OpBuilder &builder, ttg::MemDescIndexOp index,
                         Value slot0, Value slot1) {
  APInt constantIndex;
  if (matchPattern(index.getIndex(), m_ConstantInt(&constantIndex))) {
    int64_t slot = constantIndex.getSExtValue();
    assert((slot >= 0 && slot < kNumLDSBuffers) &&
           "two-slot buffer index must be zero or one");
    return slot == 0 ? slot0 : slot1;
  }

  Location loc = index.getLoc();
  auto indexType = cast<IntegerType>(index.getIndex().getType());
  Value zero = arith::ConstantOp::create(builder, loc,
                                         builder.getIntegerAttr(indexType, 0));
  Value isZero = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       index.getIndex(), zero);
  auto ifOp = scf::IfOp::create(builder, loc, index.getType(), isZero, true);
  OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
  scf::YieldOp::create(thenBuilder, loc, slot0);
  OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
  scf::YieldOp::create(elseBuilder, loc, slot1);
  return ifOp.getResult(0);
}

// Return whether a memdesc is carried from one specific combined allocation.
// Pipeline epilogues may expose it directly, through memdesc_index/select, or
// as an scf.for result, so SSA identity after materialization is insufficient.
bool isDerivedFromBuffer(Value value, Value buffer,
                         llvm::SmallDenseSet<Value> &visited) {
  if (value == buffer)
    return true;
  if (!visited.insert(value).second)
    return false;

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop || argument.getOwner() != loop.getBody() ||
        argument.getArgNumber() == 0)
      return false;
    unsigned index = argument.getArgNumber() - 1;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return isDerivedFromBuffer(loop.getInitArgs()[index], buffer, visited) ||
           isDerivedFromBuffer(yield.getOperand(index), buffer, visited);
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;
  Operation *owner = result.getOwner();
  if (auto index = dyn_cast<ttg::MemDescIndexOp>(owner))
    return isDerivedFromBuffer(index.getSrc(), buffer, visited);
  if (auto select = dyn_cast<arith::SelectOp>(owner))
    return isDerivedFromBuffer(select.getTrueValue(), buffer, visited) ||
           isDerivedFromBuffer(select.getFalseValue(), buffer, visited);
  if (auto loop = dyn_cast<scf::ForOp>(owner)) {
    unsigned index = result.getResultNumber();
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return isDerivedFromBuffer(loop.getInitArgs()[index], buffer, visited) ||
           isDerivedFromBuffer(yield.getOperand(index), buffer, visited);
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(owner)) {
    unsigned index = result.getResultNumber();
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    return isDerivedFromBuffer(thenYield.getOperand(index), buffer, visited) ||
           isDerivedFromBuffer(elseYield.getOperand(index), buffer, visited);
  }
  return false;
}

bool isDerivedFromBuffer(Value value, Value buffer) {
  llvm::SmallDenseSet<Value> visited;
  return isDerivedFromBuffer(value, buffer, visited);
}

// A generic pipeline epilogue stores its prefetched next generation before it
// loads the current generation because the ring slots are disjoint.  For an
// asymmetric single-buffered operand (Bit8 BK128 or Bit16 BK64), hoist the
// current load and bracket the overwrite with full-workgroup LDS barriers.
// The first barrier retires every wave's old-generation reads before any wave
// overwrites the common slot.  The second publishes every wave's new-generation
// writes before the later load consumes data produced by other waves.
LogicalResult makeSingleBufferEpilogueSafe(scf::ForOp loop, Value buffer) {
  ttg::LocalStoreOp nextStore;
  ttg::LocalLoadOp currentLoad;
  for (Operation *op = loop->getNextNode(); op; op = op->getNextNode()) {
    if (!nextStore) {
      if (auto store = dyn_cast<ttg::LocalStoreOp>(op);
          store && isDerivedFromBuffer(store.getDst(), buffer))
        nextStore = store;
      continue;
    }
    if (auto load = dyn_cast<ttg::LocalLoadOp>(op);
        load && isDerivedFromBuffer(load.getSrc(), buffer)) {
      currentLoad = load;
      break;
    }
  }
  if (!nextStore || !currentLoad)
    return failure();

  currentLoad->moveBefore(nextStore);
  OpBuilder builder(currentLoad);
  builder.setInsertionPointAfter(currentLoad);
  ttg::LocalBarrierOp::create(builder, currentLoad.getLoc());
  builder.setInsertionPointAfter(nextStore);
  ttg::LocalBarrierOp::create(builder, nextStore.getLoc());
  return success();
}

// Materialize each proven A/B ring allocation as static LocalAllocOps.  Normal
// double buffers receive separate slot0/slot1 allocations.  A proven
// single-buffer operand maps both logical generations to one allocation; its
// explicit read-retirement and write-publication barriers make that physical
// alias safe and visible to Membar.
LogicalResult materializeStaticDoubleBuffers(scf::ForOp loop) {
  auto plans = matchStaticDoubleBuffers(loop);
  if (failed(plans))
    return failure();

  // A two-BK main body always starts on slot0.  The first cloned BK consumes
  // slot0 and publishes slot1; the second consumes slot1 and publishes slot0.
  // Giving each slot its own LocalAllocOp makes this disjointness visible to
  // Allocation/Membar.  It therefore proves the three hot-path transitions
  // below safe without an inserted local_barrier:
  //
  //   local_load(slot0) -> local_store(slot1)
  //   local_store(slot1) -> local_load(slot0, second K half)
  //   local_load(slot0) -> local_store(slot1, B)
  //
  // A normal double-buffer epilogue still selects a slot dynamically with
  // scf.if.  A single-buffer epilogue instead uses slot0 after the explicit
  // load/barrier/store/barrier protection above. Neither case changes the
  // steady-state BlockPingpong phases.
  for (StaticBufferPlan &plan : *plans) {
    ttg::LocalAllocOp combined = plan.combinedAlloc;
    Type slotType = plan.hotIndices.front().getType();
    OpBuilder builder(combined);
    bool isSingleBuffer = combined->hasAttr(kSingleBufferAttr);
    if (isSingleBuffer &&
        failed(makeSingleBufferEpilogueSafe(loop, combined.getResult())))
      return failure();
    Value slot0 =
        ttg::LocalAllocOp::create(builder, combined.getLoc(), slotType);
    Value slot1 =
        isSingleBuffer
            ? slot0
            : ttg::LocalAllocOp::create(builder, combined.getLoc(), slotType)
                  .getResult();

    DenseMap<Operation *, Value> hotReplacements;
    hotReplacements[plan.hotIndices[0]] = slot1;
    hotReplacements[plan.hotIndices[1]] = slot0;

    SmallVector<ttg::MemDescIndexOp> allIndices;
    SmallVector<ttg::LocalDeallocOp> deallocs;
    for (Operation *user :
         llvm::make_early_inc_range(combined.getResult().getUsers())) {
      if (auto index = dyn_cast<ttg::MemDescIndexOp>(user))
        allIndices.push_back(index);
      else if (auto dealloc = dyn_cast<ttg::LocalDeallocOp>(user))
        deallocs.push_back(dealloc);
    }

    for (ttg::MemDescIndexOp index : allIndices) {
      Value replacement = hotReplacements.lookup(index);
      if (!replacement) {
        if (isSingleBuffer) {
          replacement = slot0;
        } else {
          builder.setInsertionPoint(index);
          replacement = selectStaticBuffer(builder, index, slot0, slot1);
        }
      }
      index.replaceAllUsesWith(replacement);
      index.erase();
    }
    for (ttg::LocalDeallocOp dealloc : deallocs) {
      builder.setInsertionPoint(dealloc);
      if (!isSingleBuffer)
        ttg::LocalDeallocOp::create(builder, dealloc.getLoc(), slot1);
      ttg::LocalDeallocOp::create(builder, dealloc.getLoc(), slot0);
      dealloc.erase();
    }
    combined.erase();
  }

  loop->setAttr(kStaticBuffersAttr, UnitAttr::get(loop.getContext()));
  return success();
}

// Follow a local-load memdesc through a loop-carried block argument and return
// the two-generation LDS allocation that owns it.
FailureOr<ttg::LocalAllocOp> findTwoSlotAllocation(Value value,
                                                   scf::ForOp loop) {
  llvm::SmallDenseSet<Value> visited;
  std::function<FailureOr<ttg::LocalAllocOp>(Value)> trace =
      [&](Value current) -> FailureOr<ttg::LocalAllocOp> {
    if (!visited.insert(current).second)
      return failure();

    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      auto sourceType = dyn_cast<ttg::MemDescType>(index.getSrc().getType());
      auto alloc = index.getSrc().getDefiningOp<ttg::LocalAllocOp>();
      if (sourceType && alloc && !sourceType.getShape().empty() &&
          sourceType.getShape().front() == kNumLDSBuffers)
        return alloc;
      return failure();
    }

    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      if (blockArg.getOwner() != loop.getBody())
        return failure();
      if (auto tiedInit = loop.getTiedLoopInit(blockArg))
        return trace(tiedInit->get());
    }
    return failure();
  };
  return trace(value);
}

// Follow a loop-carried value through one or more scf.yield/block-argument
// stages until it reaches `source`.  The current three-stage schedule normally
// carries a global-load result directly to the next iteration's local_store;
// accepting a chain keeps this verifier robust to canonical pipeline forms.
bool isLoopCarriedFrom(BlockArgument argument, Value source, scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  llvm::SmallDenseSet<Value> visited;
  Value current = argument;
  while (auto blockArg = dyn_cast<BlockArgument>(current)) {
    if (!visited.insert(blockArg).second ||
        blockArg.getOwner() != loop.getBody() || blockArg.getArgNumber() == 0)
      return false;
    unsigned iterArgIndex = blockArg.getArgNumber() - 1;
    if (iterArgIndex >= yield.getNumOperands())
      return false;
    current = yield.getOperand(iterArgIndex);
    if (current == source)
      return true;
  }
  return false;
}

// Verify the exact loop-carried chains produced by AMD Pipeline:
//
//   globalLoad -> scf.yield -> prefetch block argument(s)
//              -> local_store source
//   local_store destination -> scf.yield -> current memdesc block argument
//                           -> local_load source
//
// Shape equality alone is insufficient here because moving the wrong load or
// store across a phase barrier would invalidate the pingpong schedule.
bool matchesExpandedPipelineChain(tt::LoadOp globalLoad,
                                  ttg::LocalStoreOp localStore,
                                  ttg::LocalLoadOp localLoad, scf::ForOp loop) {
  auto currentBufferArg = dyn_cast<BlockArgument>(localLoad.getSrc());
  if (!currentBufferArg || currentBufferArg.getOwner() != loop.getBody() ||
      currentBufferArg.getArgNumber() == 0)
    return false;

  auto prefetchArg = dyn_cast<BlockArgument>(localStore.getSrc());
  if (!prefetchArg || prefetchArg.getOwner() != loop.getBody() ||
      prefetchArg.getArgNumber() == 0)
    return false;

  return isLoopCarriedFrom(prefetchArg, globalLoad.getResult(), loop) &&
         isLoopCarriedFrom(currentBufferArg, localStore.getDst(), loop);
}

struct ExpandedOperandChain {
  tt::LoadOp globalLoad;
  ttg::LocalStoreOp localStore;
};

// Recover one operand's global-load/local-store chain from the local load used
// by the dot.  Def-use validation, rather than operation order, makes the
// rematch insensitive to additional register-pipelined loads such as block
// scales.  Uniqueness is required before any operation is moved.
FailureOr<ExpandedOperandChain> findExpandedOperandChain(
    ttg::LocalLoadOp localLoad, ArrayRef<int64_t> expectedShape,
    ArrayRef<tt::LoadOp> globalLoads, ArrayRef<ttg::LocalStoreOp> localStores,
    scf::ForOp loop) {
  std::optional<ExpandedOperandChain> match;
  for (ttg::LocalStoreOp localStore : localStores) {
    if (!hasShape(localStore.getSrc().getType(), expectedShape))
      continue;
    for (tt::LoadOp globalLoad : globalLoads) {
      if (!hasShape(globalLoad.getType(), expectedShape) ||
          !matchesExpandedPipelineChain(globalLoad, localStore, localLoad,
                                        loop))
        continue;
      if (match)
        return failure();
      match = ExpandedOperandChain{globalLoad, localStore};
    }
  }
  if (!match)
    return failure();
  return *match;
}

// Rematch a prepared candidate after AMD Pipeline has expanded each BufferLoad
// into global load -> local store -> loop-carried memdesc -> local load.  This
// establishes the exact A/B operations that the HCU phase scheduler may move.
FailureOr<ExpandedCandidate> matchExpandedCandidate(scf::ForOp loop) {
  if (!hasRequiredWaveConfiguration(loop) || !loop->hasAttr(kCandidateAttr) ||
      loop->hasAttr(kAppliedAttr))
    return failure();

  SmallVector<tt::DotOp> dots;
  SmallVector<tt::DotScaledOp> scaledDots;
  SmallVector<tt::LoadOp> globalLoads;
  SmallVector<ttg::LocalLoadOp> localLoads;
  SmallVector<ttg::LocalStoreOp> localStores;
  bool hasUnsupportedCopy = false;
  bool hasNestedControl = false;

  loop.walk([&](Operation *op) {
    if (isa<tt::MatrixLoadOp, tta::MatrixLoadToLocalOp,
            ttg::AsyncCopyGlobalToLocalOp>(op))
      hasUnsupportedCopy = true;
    if (auto dot = dyn_cast<tt::DotOp>(op))
      dots.push_back(dot);
    if (auto dot = dyn_cast<tt::DotScaledOp>(op))
      scaledDots.push_back(dot);
    if (auto load = dyn_cast<tt::LoadOp>(op))
      globalLoads.push_back(load);
    if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
      if (auto source = dyn_cast<BlockArgument>(load.getSrc());
          source && source.getOwner() == loop.getBody())
        localLoads.push_back(load);
    }
    if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
      localStores.push_back(store);
    if (op != loop && op->getNumRegions() != 0)
      hasNestedControl = true;
  });

  if (hasUnsupportedCopy || hasNestedControl || dots.size() != 1 ||
      !scaledDots.empty() || globalLoads.size() < kNumOperands ||
      localLoads.size() != kNumOperands || localStores.size() != kNumOperands)
    return failure();

  tt::DotOp dot = dots.front();
  auto supportedTile = getSupportedTile(dot);
  if (failed(supportedTile))
    return failure();
  auto [blockM, blockN, blockK] = *supportedTile;

  auto localA = dot.getA().getDefiningOp<ttg::LocalLoadOp>();
  auto localB = dot.getB().getDefiningOp<ttg::LocalLoadOp>();
  if (!localA || !localB || !localA->hasOneUse() || !localB->hasOneUse())
    return failure();
  auto allocA = findTwoSlotAllocation(localA.getSrc(), loop);
  auto allocB = findTwoSlotAllocation(localB.getSrc(), loop);
  if (failed(allocA) || failed(allocB) || *allocA == *allocB)
    return failure();

  auto chainA = findExpandedOperandChain(localA, {blockM, blockK}, globalLoads,
                                         localStores, loop);
  auto chainB = findExpandedOperandChain(localB, {blockK, blockN}, globalLoads,
                                         localStores, loop);
  if (failed(chainA) || failed(chainB) ||
      chainA->globalLoad == chainB->globalLoad ||
      chainA->localStore == chainB->localStore)
    return failure();

  return ExpandedCandidate{loop,
                           dot,
                           chainA->globalLoad,
                           chainB->globalLoad,
                           localA,
                           localB,
                           chainA->localStore,
                           chainB->localStore,
                           *allocA,
                           *allocB,
                           blockM,
                           blockN,
                           blockK};
}

// Reorder one post-pipeline BufferLoad GEMM loop into HCU's two- or four-phase
// schedule, including optional BK/2 dot slicing, asymmetric wave
// synchronization, and the late 2K-unroll marker consumed before static-buffer
// finalization.
class BufferLoadPingponger {
public:
  explicit BufferLoadPingponger(ExpandedCandidate candidate)
      : candidate(candidate) {}

  LogicalResult run();

private:
  ExpandedCandidate candidate;
  SmallVector<SmallVector<Operation *, kNumKHalves>, kNumOperands> subviews;
  SmallVector<SmallVector<Operation *, kNumKHalves>, kNumOperands> slicedLoads;
  SmallVector<Operation *, kNumKHalves> slicedDots;
  Operation *lastInserted = nullptr;
  Value waveLowPredicate;
  Value waveHighPredicate;
  int32_t kWidth = 0;

  LogicalResult validateSlicing();
  LogicalResult sliceDot(OpBuilder &builder);
  LogicalResult createOperandSlices(OpBuilder &builder, Value operand,
                                    unsigned operandIndex,
                                    Attribute dotEncoding);
  LogicalResult moveWithBackwardSlice(Operation *op);
  bool shouldPreloadSecondBHalf();
  void append(Operation *op);
  void appendOperandSlice(unsigned operandIndex, unsigned half);
  void appendOperandHalf(unsigned half);
  void appendControlBarrier(OpBuilder &builder, Location loc);
  void appendLocalMemoryBarrier(OpBuilder &builder, Location loc);
  void appendFullBarrier(OpBuilder &builder, Location loc);
  void appendDotCluster(OpBuilder &builder, Operation *dot, Location loc);
  void appendPredicatedStore(OpBuilder &builder, Location loc, Value predicate,
                             Operation *store);
  LogicalResult scheduleSingleAPhases(OpBuilder &builder, Location loc);
  LogicalResult scheduleSingleBPhases(OpBuilder &builder, Location loc);
  LogicalResult scheduleBit8Phases(OpBuilder &builder, Location loc,
                                   bool usePanelB);
  LogicalResult scheduleBit8WholeTilePhases(OpBuilder &builder, Location loc);
  FailureOr<Operation *> findAccumulatorUpdateRoot(Operation *dot);
  LogicalResult moveAccumulatorUpdateAfterDot(Operation *dot);
  LogicalResult scheduleBit16Phases(OpBuilder &builder, Location loc,
                                    bool preloadBHalf1);
  bool shouldIssueBit8BFirst() const;
  void addAsymmetricSync(OpBuilder &builder, Location loc);
};

// Validate the post-pipeline tensor/memdesc contract required to split both
// operands along K and rebuild two legal dot-operand encodings.
LogicalResult BufferLoadPingponger::validateSlicing() {
  auto aType = dyn_cast<RankedTensorType>(candidate.dot.getA().getType());
  auto bType = dyn_cast<RankedTensorType>(candidate.dot.getB().getType());
  if (!aType || !bType || aType.getRank() != 2 || bType.getRank() != 2 ||
      aType.getShape()[1] != candidate.blockK ||
      bType.getShape()[0] != candidate.blockK)
    return failure();

  auto aEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(aType.getEncoding());
  auto bEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(bType.getEncoding());
  if (!aEncoding || !bEncoding ||
      aEncoding.getKWidth() != bEncoding.getKWidth())
    return failure();

  auto aMemDesc =
      dyn_cast<ttg::MemDescType>(candidate.localA.getSrc().getType());
  auto bMemDesc =
      dyn_cast<ttg::MemDescType>(candidate.localB.getSrc().getType());
  if (!aMemDesc || !bMemDesc || aMemDesc.getRank() != 2 ||
      bMemDesc.getRank() != 2)
    return failure();

  return success();
}

// Create the two BK/2 memdesc subslices and LocalLoadOps for one dot operand;
// operandIndex selects A's K dimension (1) or B's K dimension (0).
LogicalResult BufferLoadPingponger::createOperandSlices(OpBuilder &builder,
                                                        Value operand,
                                                        unsigned operandIndex,
                                                        Attribute dotEncoding) {
  auto localLoad = operand.getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad)
    return failure();

  auto memDescType = dyn_cast<ttg::MemDescType>(localLoad.getSrc().getType());
  if (!memDescType)
    return failure();

  SmallVector<int64_t> shape(memDescType.getShape());
  unsigned kDimension = operandIndex == kOperandA ? 1 : 0;
  shape[kDimension] = candidate.blockK / kNumKHalves;
  auto sliceMemDescType = ttg::MemDescType::get(
      shape, memDescType.getElementType(), memDescType.getEncoding(),
      memDescType.getMemorySpace(), memDescType.getMutableMemory(),
      memDescType.getAllocShape());
  auto dotOperandEncoding = ttg::DotOperandEncodingAttr::get(
      builder.getContext(), operandIndex, dotEncoding, kWidth);
  auto sliceTensorType = RankedTensorType::get(
      shape, memDescType.getElementType(), dotOperandEncoding);

  SmallVector<Operation *, kNumKHalves> operandSubviews;
  SmallVector<Operation *, kNumKHalves> operandLoads;
  for (unsigned half = 0; half < kNumKHalves; ++half) {
    SmallVector<int32_t> offsets(2, 0);
    offsets[kDimension] = half * (candidate.blockK / kNumKHalves);
    auto subview = ttg::MemDescSubsliceOp::create(builder, localLoad.getLoc(),
                                                  sliceMemDescType,
                                                  localLoad.getSrc(), offsets);
    auto sliceLoad = ttg::LocalLoadOp::create(
        builder, localLoad.getLoc(), sliceTensorType, subview.getResult());
    operandSubviews.push_back(subview);
    operandLoads.push_back(sliceLoad);
  }
  subviews.push_back(std::move(operandSubviews));
  slicedLoads.push_back(std::move(operandLoads));
  return success();
}

// Replace the original BK dot with two accumulator-chained BK/2 dots and
// replace its full-width A/B LocalLoadOps with the operand slices above.
LogicalResult BufferLoadPingponger::sliceDot(OpBuilder &builder) {
  if (failed(validateSlicing()))
    return failure();

  auto aEncoding = cast<ttg::DotOperandEncodingAttr>(
      cast<RankedTensorType>(candidate.dot.getA().getType()).getEncoding());
  kWidth = aEncoding.getKWidth();

  builder.setInsertionPointAfter(candidate.globalA);
  Attribute dotEncoding = candidate.dot.getType().getEncoding();
  if (failed(createOperandSlices(builder, candidate.dot.getA(), kOperandA,
                                 dotEncoding)) ||
      failed(createOperandSlices(builder, candidate.dot.getB(), kOperandB,
                                 dotEncoding)))
    return failure();

  Operation *previousDot = candidate.dot;
  for (unsigned half = 0; half < kNumKHalves; ++half) {
    IRMapping mapping;
    mapping.map(candidate.dot.getA(),
                slicedLoads[kOperandA][half]->getResult(0));
    mapping.map(candidate.dot.getB(),
                slicedLoads[kOperandB][half]->getResult(0));
    if (half != kFirstKHalf)
      mapping.map(candidate.dot.getC(), previousDot->getResult(0));
    Operation *newDot = builder.clone(*candidate.dot, mapping);
    slicedDots.push_back(newDot);
    previousDot = newDot;
  }

  candidate.dot->replaceAllUsesWith(previousDot->getResults());
  candidate.dot.erase();
  candidate.localA.erase();
  candidate.localB.erase();
  return success();
}

void BufferLoadPingponger::append(Operation *op) {
  assert(lastInserted && "phase insertion point must be initialized");
  op->moveAfter(lastInserted);
  lastInserted = op;
}

bool BufferLoadPingponger::shouldPreloadSecondBHalf() {
  auto memDesc =
      dyn_cast<ttg::MemDescType>(candidate.localB.getSrc().getType());
  // Keep the common four-phase schedule independent of the source-level
  // row-row/row-col spelling.  Only the physical panel-major ds_read_m
  // contract changes the profitable placement of B half1.
  return tt::HCU::isDsReadMPanelMemDesc(memDesc);
}

// Move an operation to the current phase insertion point together with any
// same-block backward dependencies that have not already been scheduled.
// Reject moves that would place an operation after an existing earlier user.
LogicalResult BufferLoadPingponger::moveWithBackwardSlice(Operation *op) {
  assert(lastInserted && "phase insertion point must be initialized");
  if (op->getBlock() != lastInserted->getBlock())
    return failure();

  Operation *checked = lastInserted;
  if (lastInserted->isBeforeInBlock(op)) {
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *dependency) {
      return dependency->getBlock() == checked->getBlock() &&
             checked->isBeforeInBlock(dependency);
    };
    (void)getBackwardSlice(op, &backwardSlice, options);
    for (Operation *dependency : backwardSlice)
      append(dependency);
    append(op);
    return success();
  }

  bool hasUserBeforeInsertion =
      llvm::any_of(op->getUsers(), [&](Operation *user) {
        return user->getBlock() == checked->getBlock() &&
               (user == checked || user->isBeforeInBlock(checked));
      });
  if (hasUserBeforeInsertion)
    return failure();
  append(op);
  return success();
}

void BufferLoadPingponger::appendOperandSlice(unsigned operandIndex,
                                              unsigned half) {
  append(subviews[operandIndex][half]);
  append(slicedLoads[operandIndex][half]);
}

void BufferLoadPingponger::appendOperandHalf(unsigned half) {
  appendOperandSlice(kOperandA, half);
  appendOperandSlice(kOperandB, half);
}

void BufferLoadPingponger::appendControlBarrier(OpBuilder &builder,
                                                Location loc) {
  // A control barrier advances the two wave groups by one BlockPingpong phase.
  // It is intentionally not gpu.barrier: M0 has only produced A(k+1), while
  // D0/D1 have produced no LDS data.  Treating these boundaries as full memory
  // barriers makes MembarAnalysis drain outstanding VMEM/DS work unnecessarily.
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  append(ROCDL::SBarrierOp::create(builder, loc));
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
}

void BufferLoadPingponger::appendLocalMemoryBarrier(OpBuilder &builder,
                                                    Location loc) {
  // Unlike a control-only phase boundary, this boundary protects a physical
  // single-buffer overwrite. LocalBarrier is visible to MembarAnalysis and on
  // CDNA lowers exactly to an LDS wait followed by a whole-CTA s_barrier; it
  // deliberately does not drain unrelated VMEM operations.
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  append(ttg::LocalBarrierOp::create(builder, loc));
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
}

void BufferLoadPingponger::appendFullBarrier(OpBuilder &builder, Location loc) {
  // M1 completes B(k+1), so A+B(k+1) is now a complete LDS generation.  This
  // is the only steady-state boundary that publishes memory and permits the
  // peer wave group to consume the next slot.
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  append(gpu::BarrierOp::create(builder, loc));
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
}

void BufferLoadPingponger::appendDotCluster(OpBuilder &builder, Operation *dot,
                                            Location loc) {
  // HCU cannot co-issue MMAC and VALU.  Memory phases stay at priority 1;
  // lowering the MMAC wave to 0 gives its peer a chance to issue address/DS
  // work.  Existing HCU dot lowering moves the immediately preceding setprio
  // after the first MMAC, so one MMAC enters at priority 1 and the remaining
  // cluster runs at priority 0.  Keeping that established behavior avoids a
  // global lowering change; measurement showed no material throughput loss.
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  append(ROCDL::SetPrioOp::create(builder, loc, 0));
  append(dot);
  append(ROCDL::SetPrioOp::create(builder, loc, 1));
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
}

void BufferLoadPingponger::appendPredicatedStore(OpBuilder &builder,
                                                 Location loc, Value predicate,
                                                 Operation *store) {
  assert(predicate && "wave-group predicate must be initialized");
  assert(store && "predicated store must be initialized");
  builder.setInsertionPointAfter(lastInserted);
  auto storeIf = scf::IfOp::create(builder, loc, TypeRange{}, predicate,
                                   /*withElseRegion=*/false);
  OpBuilder thenBuilder = storeIf.getThenBodyBuilder();
  thenBuilder.clone(*store);
  append(storeIf);
}

void BufferLoadPingponger::addAsymmetricSync(OpBuilder &builder, Location loc) {
  // Divide eight 64-lane waves into two groups of four.  thread_id_x is
  // lane-varying at the ISA level even though thread_id_x >> 6 is wave-uniform;
  // readfirstlane makes that uniformity explicit before CondBarrier lowering.
  // The resulting scalar branch keeps s_barrier outside EXEC-masked control
  // flow without changing the generic AMD CondBarrier conversion.  Each SIMD
  // receives one wave from each group.  Holding WaveLow before the loop lets
  // WaveHigh lead by exactly one phase; holding WaveHigh afterwards lets
  // WaveLow drain its final phase before the workgroup reconverges.
  builder.setInsertionPoint(candidate.loop);
  gpu::BarrierOp::create(builder, loc);
  ROCDL::SetPrioOp::create(builder, loc, 1);

  auto i32Type = builder.getI32Type();
  Value threadId = ROCDL::ThreadIdXOp::create(builder, loc, i32Type);
  Value waveId = arith::ShRUIOp::create(
      builder, loc, threadId,
      arith::ConstantIntOp::create(builder, loc, kWaveIdShift, 32));
  waveId = LLVM::createLLVMIntrinsicCallOp(
               builder, loc, "llvm.amdgcn.readfirstlane", {i32Type}, {waveId})
               .getResult(0);
  // On current BW1000/BW1100 hardware an eight-wave CTA is placed as
  //   SIMD0: warp0/warp4, SIMD1: warp1/warp5,
  //   SIMD2: warp2/warp6, SIMD3: warp3/warp7.
  // Splitting warp0-3 from warp4-7 therefore gives every SIMD one wave from
  // each phase. Data safety remains a separate buffer/barrier contract: the
  // asymmetric physical-single paths load the complete aliased operand
  // before a full-CTA local barrier permits any wave to overwrite it.
  Value groupSize =
      arith::ConstantIntOp::create(builder, loc, kWaveGroupSize, 32);
  waveLowPredicate = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, waveId, groupSize);
  waveHighPredicate = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::uge, waveId, groupSize);
  tta::CondBarrierOp::create(builder, loc, waveLowPredicate);

  builder.setInsertionPointAfter(candidate.loop);
  tta::CondBarrierOp::create(builder, loc, waveHighPredicate);
}

bool BufferLoadPingponger::shouldIssueBit8BFirst() const {
  // One A half contains BM*BK/2 elements and one B half contains BK/2*BN.
  // Start the heavier operand first so its LDS latency has at least as much
  // cover. Prefer B on ties because B1 is also carried across D0. This derives
  // the order from the tile instead of recognizing a tuned matrix size.
  return candidate.blockN >= candidate.blockM;
}

// BM256xBN128 physical-single A path.  It is deliberately separate from the
// mirrored single-B path: M1 overwrites A unconditionally, so M0 itself must
// drain every old-A read before the peer can enter M1.
LogicalResult BufferLoadPingponger::scheduleSingleAPhases(OpBuilder &builder,
                                                          Location loc) {
  appendOperandSlice(kOperandA, kFirstKHalf);
  appendOperandSlice(kOperandB, kFirstKHalf);
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  if (failed(moveWithBackwardSlice(candidate.globalA)))
    return failure();
  appendOperandSlice(kOperandA, kSecondKHalf);
  appendLocalMemoryBarrier(builder, loc);

  if (failed(moveWithBackwardSlice(candidate.globalB)))
    return failure();
  appendDotCluster(builder, slicedDots[kFirstKHalf], loc);
  appendLocalMemoryBarrier(builder, loc);

  appendOperandSlice(kOperandB, kSecondKHalf);
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  if (failed(moveWithBackwardSlice(candidate.storeB)) ||
      failed(moveWithBackwardSlice(candidate.storeA)))
    return failure();
  appendFullBarrier(builder, loc);

  appendDotCluster(builder, slicedDots[kSecondKHalf], loc);
  if (failed(moveAccumulatorUpdateAfterDot(slicedDots[kSecondKHalf])))
    return failure();
  appendControlBarrier(builder, loc);
  return success();
}

// BM128xBN256 physical-single B path.  B stores are split between Low-M1 and
// High-D1.  Both groups cross D0's LocalBarrier before either store can run,
// so M0 needs only a control rendezvous; the final D1/M1 rendezvous publishes
// the collectively written generation.
LogicalResult BufferLoadPingponger::scheduleSingleBPhases(OpBuilder &builder,
                                                          Location loc) {
  appendOperandSlice(kOperandB, kFirstKHalf);
  appendOperandSlice(kOperandA, kFirstKHalf);
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  if (failed(moveWithBackwardSlice(candidate.globalB)))
    return failure();
  appendOperandSlice(kOperandB, kSecondKHalf);
  appendControlBarrier(builder, loc);

  if (failed(moveWithBackwardSlice(candidate.globalA)))
    return failure();
  appendDotCluster(builder, slicedDots[kFirstKHalf], loc);
  appendLocalMemoryBarrier(builder, loc);

  if (failed(moveWithBackwardSlice(candidate.storeA)))
    return failure();
  appendPredicatedStore(builder, loc, waveLowPredicate, candidate.storeB);
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  appendOperandSlice(kOperandA, kSecondKHalf);
  appendFullBarrier(builder, loc);

  appendDotCluster(builder, slicedDots[kSecondKHalf], loc);
  if (failed(moveAccumulatorUpdateAfterDot(slicedDots[kSecondKHalf])))
    return failure();
  appendPredicatedStore(builder, loc, waveHighPredicate, candidate.storeB);
  appendLocalMemoryBarrier(builder, loc);
  candidate.storeB.erase();
  return success();
}

// Ordinary Bit8 double-buffered path.  Physical-single schedules are kept in
// the two functions above so changes to one overwrite contract cannot silently
// perturb the other or the established two-A/two-B family.
LogicalResult BufferLoadPingponger::scheduleBit8Phases(OpBuilder &builder,
                                                       Location loc,
                                                       bool usePanelB) {
  bool useDoubleBufferLatencySchedule = !usePanelB && candidate.blockM == 128 &&
                                        candidate.blockN == 128 &&
                                        candidate.blockK == 128;
  if (shouldIssueBit8BFirst()) {
    appendOperandSlice(kOperandB, kFirstKHalf);
    appendOperandSlice(kOperandA, kFirstKHalf);
  } else {
    appendOperandSlice(kOperandA, kFirstKHalf);
    appendOperandSlice(kOperandB, kFirstKHalf);
  }
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  if (usePanelB) {
    if (failed(moveWithBackwardSlice(candidate.globalB)) ||
        failed(moveWithBackwardSlice(candidate.globalA)))
      return failure();
    appendOperandSlice(kOperandB, kSecondKHalf);
    if (failed(moveWithBackwardSlice(candidate.storeA)))
      return failure();
  } else {
    if (failed(moveWithBackwardSlice(candidate.globalA)) ||
        failed(moveWithBackwardSlice(candidate.globalB)))
      return failure();
    appendOperandSlice(kOperandB, kSecondKHalf);
  }
  appendControlBarrier(builder, loc);

  appendDotCluster(builder, slicedDots[kFirstKHalf], loc);
  if (useDoubleBufferLatencySchedule &&
      failed(moveWithBackwardSlice(candidate.storeA)))
    return failure();
  appendControlBarrier(builder, loc);

  appendOperandSlice(kOperandA, kSecondKHalf);
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  if (usePanelB) {
    if (failed(moveWithBackwardSlice(candidate.storeB)))
      return failure();
  } else if ((!useDoubleBufferLatencySchedule &&
              failed(moveWithBackwardSlice(candidate.storeA))) ||
             failed(moveWithBackwardSlice(candidate.storeB))) {
    return failure();
  }
  appendFullBarrier(builder, loc);

  appendDotCluster(builder, slicedDots[kSecondKHalf], loc);
  if (failed(moveAccumulatorUpdateAfterDot(slicedDots[kSecondKHalf])))
    return failure();
  appendControlBarrier(builder, loc);
  return success();
}

// Pair one complete-BK memory phase with one complete-BK dot/update phase so
// post-dot reductions do not pay four rendezvous without being splittable.
// Physical-single BK128 operands use the split four-phase builder above,
// which reads their complete old tile before permitting an overwrite.
LogicalResult
BufferLoadPingponger::scheduleBit8WholeTilePhases(OpBuilder &builder,
                                                  Location loc) {
  if (shouldIssueBit8BFirst()) {
    if (failed(moveWithBackwardSlice(candidate.localB)) ||
        failed(moveWithBackwardSlice(candidate.localA)))
      return failure();
  } else if (failed(moveWithBackwardSlice(candidate.localA)) ||
             failed(moveWithBackwardSlice(candidate.localB))) {
    return failure();
  }

  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  if (failed(moveWithBackwardSlice(candidate.globalA)) ||
      failed(moveWithBackwardSlice(candidate.globalB)))
    return failure();
  if (failed(moveWithBackwardSlice(candidate.storeA)) ||
      failed(moveWithBackwardSlice(candidate.storeB)))
    return failure();
  appendFullBarrier(builder, loc);

  appendDotCluster(builder, candidate.dot, loc);
  if (failed(moveAccumulatorUpdateAfterDot(candidate.dot)))
    return failure();
  appendControlBarrier(builder, loc);
  return success();
}

// Keep the current BK's register-only accumulator update in the compute phase.
// This covers INT32-to-FP32 accumulation and block-scale multiplication without
// recognizing a source dtype or quantization spelling.  The unique loop yield
// that depends on the dot is the semantic boundary; pointer and future-scale
// pipeline values are independent and remain in the memory phase.
FailureOr<Operation *>
BufferLoadPingponger::findAccumulatorUpdateRoot(Operation *dot) {
  auto yield =
      dyn_cast<scf::YieldOp>(candidate.loop.getBody()->getTerminator());
  if (!yield)
    return failure();

  Operation *updateRoot = nullptr;
  for (Value yielded : yield.getOperands()) {
    Operation *root = yielded.getDefiningOp();
    if (!root || root->getBlock() != candidate.loop.getBody())
      continue;
    if (root == dot) {
      if (updateRoot)
        return failure();
      updateRoot = root;
      continue;
    }

    SetVector<Operation *> dependencies;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *dependency) {
      return dependency->getBlock() == candidate.loop.getBody();
    };
    (void)getBackwardSlice(root, &dependencies, options);
    if (!dependencies.contains(dot))
      continue;
    if (updateRoot)
      return failure();
    updateRoot = root;
  }

  if (!updateRoot)
    return failure();
  return updateRoot;
}

LogicalResult
BufferLoadPingponger::moveAccumulatorUpdateAfterDot(Operation *dot) {
  auto updateRoot = findAccumulatorUpdateRoot(dot);
  if (failed(updateRoot))
    return failure();
  if (*updateRoot == dot)
    return success();

  // The compute phase may absorb register-only arithmetic, but must not pull
  // a late scale/global/local read across the phase boundary.  Loads already
  // issued in the memory phase remain valid dependencies of the update.
  SetVector<Operation *> dependencies;
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  options.filter = [&](Operation *dependency) {
    return dependency->getBlock() == candidate.loop.getBody();
  };
  (void)getBackwardSlice(*updateRoot, &dependencies, options);
  if (llvm::any_of(dependencies, [&](Operation *dependency) {
        return !isMemoryEffectFree(dependency) &&
               lastInserted->isBeforeInBlock(dependency);
      }))
    return failure();
  return moveWithBackwardSlice(*updateRoot);
}

LogicalResult BufferLoadPingponger::scheduleBit16Phases(OpBuilder &builder,
                                                        Location loc,
                                                        bool preloadBHalf1) {
  // M0: retain the BF16/FP16 order exactly.  The old A generation is stored
  // before future A is issued; panel B half1 is then preloaded across D0.
  appendOperandHalf(kFirstKHalf);
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  if (failed(moveWithBackwardSlice(candidate.storeA)) ||
      failed(moveWithBackwardSlice(candidate.globalA)))
    return failure();
  if (preloadBHalf1)
    appendOperandSlice(kOperandB, kSecondKHalf);
  appendControlBarrier(builder, loc);

  appendDotCluster(builder, slicedDots[kFirstKHalf], loc);
  appendControlBarrier(builder, loc);

  // M1: read half1, then publish/issue B at the only full memory barrier.
  appendOperandSlice(kOperandA, kSecondKHalf);
  if (!preloadBHalf1)
    appendOperandSlice(kOperandB, kSecondKHalf);
  append(ROCDL::SchedBarrier::create(builder, loc, 0));
  if (failed(moveWithBackwardSlice(candidate.storeB)) ||
      failed(moveWithBackwardSlice(candidate.globalB)))
    return failure();
  appendFullBarrier(builder, loc);

  appendDotCluster(builder, slicedDots[kSecondKHalf], loc);
  if (failed(moveAccumulatorUpdateAfterDot(slicedDots[kSecondKHalf])))
    return failure();
  appendControlBarrier(builder, loc);
  return success();
}

// Build the selected two- or four-phase body transactionally on a cloned
// candidate.  On success, attach the common late-unroll/finalization state.
LogicalResult BufferLoadPingponger::run() {
  OpBuilder builder(candidate.loop);
  Location loc = candidate.dot.getLoc();
  auto aType = cast<RankedTensorType>(candidate.dot.getA().getType());
  unsigned storageBitWidth = getStorageBitWidth(aType);
  bool isBit8Pipeline = hasElementBitWidth(aType.getElementType(), kBit8Width);
  bool useSingleA = usesSingleABuffer(candidate.blockM, candidate.blockN,
                                      candidate.blockK, storageBitWidth);
  bool useSingleB = usesSingleBBuffer(candidate.blockM, candidate.blockN,
                                      candidate.blockK, storageBitWidth);
  auto accumulatorUpdateRoot =
      findAccumulatorUpdateRoot(candidate.dot.getOperation());
  bool hasPostDotAccumulatorUpdate = succeeded(accumulatorUpdateRoot) &&
                                     *accumulatorUpdateRoot != candidate.dot;
  bool useWholeTileBit8Phases = isBit8Pipeline && hasPostDotAccumulatorUpdate;
  bool preloadBHalf1 = shouldPreloadSecondBHalf();
  if (!useWholeTileBit8Phases && failed(sliceDot(builder)))
    return failure();

  Operation *anchor = candidate.globalA->getPrevNode();
  if (!anchor)
    return failure();
  lastInserted = anchor;
  addAsymmetricSync(builder, loc);
  builder.setInsertionPointAfter(anchor);

  if (useWholeTileBit8Phases) {
    if (failed(scheduleBit8WholeTilePhases(builder, loc)))
      return failure();
  } else if (useSingleA) {
    if (failed(scheduleSingleAPhases(builder, loc)))
      return failure();
  } else if (useSingleB) {
    if (failed(scheduleSingleBPhases(builder, loc)))
      return failure();
  } else if (isBit8Pipeline) {
    if (failed(scheduleBit8Phases(builder, loc, preloadBHalf1)))
      return failure();
  } else if (failed(scheduleBit16Phases(builder, loc, preloadBHalf1))) {
    return failure();
  }

  candidate.loop->setAttr(kLoopUnrollFactorAttr,
                          builder.getI32IntegerAttr(kLateUnrollFactor));
  candidate.loop->removeAttr(kCandidateAttr);
  candidate.loop->removeAttr(kLocalStoreStageAttr);
  candidate.loop->removeAttr(kNumBuffersAttr);
  candidate.loop->setAttr(kAppliedAttr, builder.getUnitAttr());
  candidate.allocA->setAttr(kPingpongBufferAttr, builder.getUnitAttr());
  candidate.allocB->setAttr(kPingpongBufferAttr, builder.getUnitAttr());
  if (useSingleA) {
    // A is protected by the full-CTA read-before-overwrite boundary.  B does
    // not alias and remains a two-slot ring.
    candidate.allocA->setAttr(kSingleBufferAttr, builder.getUnitAttr());
  } else if (useSingleB) {
    candidate.allocB->setAttr(kSingleBufferAttr, builder.getUnitAttr());
  }
  if (isBit8Pipeline) {
    if (auto function = candidate.loop->getParentOfType<tt::FuncOp>()) {
      function->setAttr(kPackBlockPingpongBit8Attr, builder.getUnitAttr());
      function->getParentOfType<ModuleOp>()->setAttr(kPackBlockPingpongBit8Attr,
                                                     builder.getUnitAttr());
    }
  }
  Operation *remarkDot = useWholeTileBit8Phases ? candidate.dot.getOperation()
                                                : slicedDots.front();
  remarkDot->emitRemark()
      << "performed HCU BufferLoad BlockPingpong scheduling\n";
  return success();
}

// Pre-pipeline pass: select legal source loops, split supported K-boundary
// tails, and attach the stage/store/buffer controls consumed by AMD Pipeline.
struct TritonHCUPrepareBlockPingpongPass
    : impl::TritonHCUPrepareBlockPingpongBase<
          TritonHCUPrepareBlockPingpongPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    triton::AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(module);
    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops) {
      auto candidate = matchOriginalCandidate(loop);
      if (failed(candidate))
        continue;

      // A loop-invariant mask (for example the MoE token-validity mask) is
      // independent of K and can be pipelined with the load unchanged.  Only
      // a loop-variant mask may describe a partial K tile and therefore needs
      // the unmasked-main plus one-shot-tail rewrite below.
      bool hasLoopVariantMask =
          llvm::any_of(candidate->loads, [&](tt::LoadOp load) {
            Value mask = load.getMask();
            return mask && !loop.isDefinedOutsideOfLoop(mask);
          });
      auto aType = cast<RankedTensorType>(candidate->dot.getA().getType());
      if (hasLoopVariantMask &&
          hasElementBitWidth(aType.getElementType(), kBit8Width) &&
          candidate->blockK == kExtendedBit8BlockK) {
        // The one-shot tail created below remains a direct global-load dot;
        // AMD Pipeline does not give it the LDS chain required by
        // sliceMaskedTailDot. Keep this uncommon BK128 non-divisible-K shape
        // on the ordinary pipeline instead of retaining a full-BK operand live
        // beside the accumulator and risking a severe VGPR/spill cliff.
        continue;
      }
      std::optional<KBoundaryTailPlan> tailPlan;
      DenseMap<Operation *, Value> effectiveMasks;
      if (hasLoopVariantMask) {
        auto plan = buildKBoundaryTailPlan(*candidate);
        if (failed(plan))
          continue;
        tailPlan = std::move(*plan);
        effectiveMasks = getMainLoopMasks(*tailPlan);
      }

      if (!hasRequiredGlobalLoadVectorization(*candidate, axisInfoAnalysis,
                                              tailPlan ? &effectiveMasks
                                                       : nullptr))
        continue;

      preserveMaskConstancyHints(*candidate, axisInfoAnalysis,
                                 tailPlan ? &effectiveMasks : nullptr);

      if (tailPlan) {
        candidate = applyKBoundaryTailPlan(*tailPlan);
        loop = candidate->loop;
      }

      // AMD LowerLoops reads these HCU-specific overrides during Pipeline.
      // Use a three-stage pipeline: stage0 issues global loads, stage1 writes
      // the preceding prefetched tile to LDS, and stage2 performs local_load +
      // dot. Compared with the former stage4/store2 schedule, removing the
      // empty VGPR-carry stage shortens global-load live ranges while retaining
      // the same two LDS generations. The later scheduler selects either the
      // M0-D0-M1-D1 or whole-BK M-D phase topology from reduction dataflow. AMD
      // already places local_load at the last stage, so HCU only overrides
      // local_store and the logical LDS generation count.  The matcher has
      // already proved that either two complete A/B generations, or one of the
      // exact asymmetric Bit8 BK128 / Bit16 BK64 single-operand mappings, fits
      // in HCU's 64 KiB LDS.
      OpBuilder builder(loop);
      loop->setAttr(tt::kNumStagesAttrName,
                    builder.getI32IntegerAttr(kRequiredStages));
      loop->setAttr(kLocalStoreStageAttr,
                    builder.getI32IntegerAttr(kLocalStoreStage));
      loop->setAttr(kNumBuffersAttr, builder.getI32IntegerAttr(kNumLDSBuffers));
      loop->setAttr(kCandidateAttr, builder.getUnitAttr());
    }
  }
};

// Post-pipeline pass: reduce masked-tail VGPR pressure and transactionally
// apply the selected HCU two- or four-phase schedule to each expanded
// candidate.
struct TritonHCUBlockPingpongPass
    : impl::TritonHCUBlockPingpongBase<TritonHCUBlockPingpongPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> maskedTails;
    module.walk([&](Operation *op) {
      if (op->hasAttr(kMaskedTailAttr))
        maskedTails.push_back(op);
    });
    // The one-BK masked tail is not phase-pipelined, but retaining a full-BK
    // A/B operand beside a large accumulator can exceed HCU's 256-VGPR limit.
    // Two ordered BK/2 dots preserve the result and lower the tail's peak live
    // range.  Canonicalization may represent the one-shot tail as scf.if, so
    // match the marker-bearing container rather than a loop.
    for (Operation *tail : maskedTails)
      (void)sliceMaskedTailDot(tail);

    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) { loops.push_back(loop); });

    bool transformed = false;
    for (scf::ForOp loop : loops) {
      if (!loop->hasAttr(kCandidateAttr))
        continue;
      auto candidate = matchExpandedCandidate(loop);
      if (failed(candidate)) {
        loop.emitError(
            "failed to rematch prepared HCU BlockPingpong candidate");
        return signalPassFailure();
      }

      // Transform a clone so a failed dependency move cannot leave a loop with
      // a sliced dot but incomplete synchronization.  Only the fully scheduled
      // clone replaces the original loop.
      OpBuilder builder(loop);
      builder.setInsertionPoint(loop);
      auto clonedLoop = cast<scf::ForOp>(builder.clone(*loop));
      auto clonedCandidate = matchExpandedCandidate(clonedLoop);
      if (failed(clonedCandidate)) {
        clonedLoop.erase();
        loop.emitError("failed to clone prepared HCU BlockPingpong candidate");
        return signalPassFailure();
      }

      BufferLoadPingponger pingponger(*clonedCandidate);
      if (failed(pingponger.run())) {
        clonedLoop.erase();
        loop.emitError(
            "failed to schedule prepared HCU BlockPingpong candidate");
        return signalPassFailure();
      }

      loop->replaceAllUsesWith(clonedLoop.getResults());
      loop.erase();
      transformed = true;
    }

    // Recompute wait counts after reordering, as in AMD BlockPingpong. Current
    // HCU candidates exclude async copies; keep this for future async support.
    if (transformed)
      tt::updateWaits(module);
  }
};

// Post-unroll pass: turn dynamic two-slot A/B rings into independent static LDS
// allocations so later alias and membar analysis can see slot disjointness.
struct TritonHCUFinalizeBlockPingpongPass
    : impl::TritonHCUFinalizeBlockPingpongBase<
          TritonHCUFinalizeBlockPingpongPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) { loops.push_back(loop); });

    // Late unroll=2 turns the ring-buffer indices into a fixed slot1/slot0
    // pair.  Materialize those slots independently so Membar can prove that
    // current-slot reads and next-slot writes do not alias.
    for (scf::ForOp loop : loops) {
      if (!requiresStaticDoubleBufferMaterialization(loop))
        continue;
      if (failed(materializeStaticDoubleBuffers(loop))) {
        loop.emitError("failed to materialize the two static HCU "
                       "BlockPingpong LDS buffers");
        signalPassFailure();
        return;
      }
    }
  }
};

// LLVM-dialect cleanup for Bit8 pipelines. Triton tensors become literal
// structs of per-thread elements during lowering. If such a struct crosses a
// CFG backedge, LLVM tends to form <4 x i8> PHIs and the AMDGPU backend then
// rebuilds every LDS-store word with long shift/or chains. Carry i32 words
// across the backedge instead and retain that representation through stores.
static bool isPackableBit8Struct(Type type) {
  auto structType = dyn_cast<LLVM::LLVMStructType>(type);
  if (!structType || structType.isIdentified() || structType.isPacked())
    return false;
  ArrayRef<Type> body = structType.getBody();
  return !body.empty() && body.size() % kBit8ElementsPerWord == 0 &&
         llvm::all_of(
             body, [](Type element) { return element.isInteger(kBit8Width); });
}

static bool hasOnlyScalarExtractUsers(BlockArgument argument) {
  return llvm::all_of(argument.getUses(), [](OpOperand &use) {
    auto extract = dyn_cast<LLVM::ExtractValueOp>(use.getOwner());
    return extract && extract.getContainer() == use.get() &&
           extract.getPosition().size() == 1;
  });
}

static Value packBit8Struct(OpBuilder &builder, Location loc, Value input,
                            LLVM::LLVMStructType packedType) {
  auto inputType = cast<LLVM::LLVMStructType>(input.getType());
  auto elementVectorType =
      VectorType::get({kBit8ElementsPerWord}, builder.getI8Type());
  TritonLLVMOpBuilder b(loc, builder);
  Value packed = LLVM::UndefOp::create(builder, loc, packedType);
  for (unsigned wordIndex = 0; wordIndex < packedType.getBody().size();
       ++wordIndex) {
    Value elements = LLVM::UndefOp::create(builder, loc, elementVectorType);
    for (unsigned lane = 0; lane < kBit8ElementsPerWord; ++lane) {
      Value element = LLVM::ExtractValueOp::create(
          builder, loc, input, wordIndex * kBit8ElementsPerWord + lane);
      elements = b.insert_element(elements, element, b.i32_val(lane));
    }
    Value word = b.bitcast(elements, builder.getI32Type());
    packed = LLVM::InsertValueOp::create(builder, loc, packedType, packed, word,
                                         wordIndex);
  }
  assert(inputType.getBody().size() ==
         packedType.getBody().size() * kBit8ElementsPerWord);
  return packed;
}

static LogicalResult tryPackLoopCarriedBlockArgument(BlockArgument argument) {
  if (!isPackableBit8Struct(argument.getType()) ||
      !hasOnlyScalarExtractUsers(argument))
    return failure();

  Block *block = argument.getOwner();
  unsigned argumentNumber = argument.getArgNumber();
  SmallVector<LLVM::BrOp> predecessors;
  for (Block *predecessor : block->getPredecessors()) {
    auto branch = dyn_cast<LLVM::BrOp>(predecessor->getTerminator());
    if (!branch || branch.getDest() != block ||
        branch.getDestOperands().size() != block->getNumArguments())
      return failure();
    predecessors.push_back(branch);
  }
  // A loop header has an entry edge and a backedge. Ordinary one-way block
  // arguments cannot become PHIs and do not need a representation change.
  if (predecessors.size() < 2)
    return failure();

  MLIRContext *context = argument.getContext();
  auto oldType = cast<LLVM::LLVMStructType>(argument.getType());
  SmallVector<Type> wordTypes(oldType.getBody().size() / kBit8ElementsPerWord,
                              IntegerType::get(context, kPackedWordWidth));
  auto packedType = LLVM::LLVMStructType::getLiteral(context, wordTypes);

  for (LLVM::BrOp branch : predecessors) {
    OpBuilder builder(branch);
    SmallVector<Value> operands(branch.getDestOperands());
    operands[argumentNumber] = packBit8Struct(
        builder, branch.getLoc(), operands[argumentNumber], packedType);
    LLVM::BrOp::create(builder, branch.getLoc(), operands, block);
    branch.erase();
  }

  SmallVector<LLVM::ExtractValueOp> extracts;
  for (OpOperand &use : argument.getUses())
    extracts.push_back(cast<LLVM::ExtractValueOp>(use.getOwner()));
  argument.setType(packedType);

  for (LLVM::ExtractValueOp extract : extracts) {
    unsigned elementIndex = extract.getPosition().front();
    OpBuilder builder(extract);
    TritonLLVMOpBuilder b(extract.getLoc(), builder);
    Value word =
        LLVM::ExtractValueOp::create(builder, extract.getLoc(), argument,
                                     elementIndex / kBit8ElementsPerWord);
    auto elementVectorType =
        VectorType::get({kBit8ElementsPerWord}, builder.getI8Type());
    Value elements = b.bitcast(word, elementVectorType);
    Value element = b.extract_element(
        elements, b.i32_val(elementIndex % kBit8ElementsPerWord));
    extract.replaceAllUsesWith(element);
    extract.erase();
  }
  return success();
}

static bool collectInsertedVectorElements(Value value,
                                          MutableArrayRef<Value> elements) {
  Value current = value;
  while (auto insert = current.getDefiningOp<LLVM::InsertElementOp>()) {
    APInt position;
    if (!matchPattern(insert.getPosition(), m_ConstantInt(&position)) ||
        position.isNegative() || position.getZExtValue() >= elements.size())
      return false;
    // Walk backwards from the final vector. The first assignment is the value
    // visible after any repeated insertion at the same lane.
    unsigned lane = position.getZExtValue();
    if (!elements[lane])
      elements[lane] = insert.getValue();
    current = insert.getVector();
  }
  return current.getDefiningOp<LLVM::UndefOp>() &&
         llvm::all_of(elements, [](Value element) { return bool(element); });
}

// Recover an original packed word from the scalar extracts introduced above.
// All elements must name the same struct field in lane order.
static Value tracePackedCarrierWord(ArrayRef<Value> elements) {
  if (elements.size() != kBit8ElementsPerWord)
    return {};

  Value container;
  unsigned wordPosition = 0;
  Value word;
  for (unsigned lane = 0; lane < kBit8ElementsPerWord; ++lane) {
    auto extractElement =
        elements[lane].getDefiningOp<LLVM::ExtractElementOp>();
    APInt elementPosition;
    if (!extractElement ||
        !matchPattern(extractElement.getPosition(),
                      m_ConstantInt(&elementPosition)) ||
        elementPosition.getZExtValue() != lane)
      return {};
    auto bitcast = extractElement.getVector().getDefiningOp<LLVM::BitcastOp>();
    auto extractWord =
        bitcast ? bitcast.getArg().getDefiningOp<LLVM::ExtractValueOp>()
                : LLVM::ExtractValueOp{};
    if (!extractWord || extractWord.getPosition().size() != 1)
      return {};
    if (lane == 0) {
      container = extractWord.getContainer();
      wordPosition = extractWord.getPosition().front();
      word = extractWord;
    } else if (extractWord.getContainer() != container ||
               extractWord.getPosition().front() != wordPosition) {
      return {};
    }
  }
  return word;
}

static LogicalResult tryPackBit8VectorStore(LLVM::StoreOp store) {
  auto vectorType = dyn_cast<VectorType>(store.getValue().getType());
  if (!vectorType || vectorType.getRank() != 1 ||
      !vectorType.getElementType().isInteger(kBit8Width) ||
      vectorType.getNumElements() % kBit8ElementsPerWord != 0)
    return failure();
  auto pointerType = dyn_cast<LLVM::LLVMPointerType>(store.getAddr().getType());
  if (!pointerType ||
      pointerType.getAddressSpace() != kSharedMemoryAddressSpace)
    return failure();

  SmallVector<Value> elements(vectorType.getNumElements());
  if (!collectInsertedVectorElements(store.getValue(), elements))
    return failure();

  OpBuilder builder(store);
  TritonLLVMOpBuilder b(store.getLoc(), builder);
  auto packedVectorType =
      VectorType::get({vectorType.getNumElements() / kBit8ElementsPerWord},
                      builder.getI32Type());
  auto elementVectorType =
      VectorType::get({kBit8ElementsPerWord}, builder.getI8Type());
  Value packed =
      LLVM::UndefOp::create(builder, store.getLoc(), packedVectorType);
  for (unsigned wordIndex = 0; wordIndex < packedVectorType.getNumElements();
       ++wordIndex) {
    ArrayRef<Value> elementGroup(elements.data() +
                                     wordIndex * kBit8ElementsPerWord,
                                 kBit8ElementsPerWord);
    Value word = tracePackedCarrierWord(elementGroup);
    if (!word) {
      Value elementVector =
          LLVM::UndefOp::create(builder, store.getLoc(), elementVectorType);
      for (unsigned lane = 0; lane < kBit8ElementsPerWord; ++lane)
        elementVector = b.insert_element(elementVector, elementGroup[lane],
                                         b.i32_val(lane));
      word = b.bitcast(elementVector, builder.getI32Type());
    }
    packed = b.insert_element(packed, word, b.i32_val(wordIndex));
  }
  store.getValueMutable().assign(packed);
  return success();
}

struct TritonHCUPackBlockPingpongBit8Pass
    : impl::TritonHCUPackBlockPingpongBit8Base<
          TritonHCUPackBlockPingpongBit8Pass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr(kPackBlockPingpongBit8Attr))
      return;
    module->removeAttr(kPackBlockPingpongBit8Attr);

    SmallVector<LLVM::LLVMFuncOp> functions;
    module.walk([&](LLVM::LLVMFuncOp function) {
      if (!function->hasAttr(kPackBlockPingpongBit8Attr))
        return;
      function->removeAttr(kPackBlockPingpongBit8Attr);
      functions.push_back(function);
    });

    for (LLVM::LLVMFuncOp function : functions) {
      SmallVector<BlockArgument> arguments;
      for (Block &block : function.getBody())
        for (BlockArgument argument : block.getArguments())
          if (isPackableBit8Struct(argument.getType()))
            arguments.push_back(argument);

      // A main pipelined loop may forward its Bit8 carrier directly to a
      // remainder loop.  Packing a successor turns the predecessor's
      // whole-struct branch use into scalar extracts.  Iterate to a fixed
      // point instead of relying on textual block order.
      bool changed;
      do {
        changed = false;
        for (BlockArgument argument : llvm::reverse(arguments))
          changed |= succeeded(tryPackLoopCarriedBlockArgument(argument));
      } while (changed);

      // Opaque pointers allow changing only the stored vector type while
      // retaining the original store's alignment and alias metadata.
      function.walk(
          [&](LLVM::StoreOp store) { (void)tryPackBit8VectorStore(store); });
    }
  }
};

} // namespace
} // namespace mlir

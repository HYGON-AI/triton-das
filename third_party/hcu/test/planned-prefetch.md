# BufferLds BlockPingpong prefetch

Enable `use_async_copy=True` and `use_block_pingpong=True` to request
BlockPingpong scheduling for eligible BufferLds producers. The pass checks
producer/consumer geometry and dependencies. Static and runtime K use the
same scheduling path; no runtime K value selects a specialized binary. Valid
forced MMAC layouts remain respected.

## Scope

The separate BufferLds transport feature owns row permutation, wrap, shared
encoding and target-specific M0 fields. This change adds prefetching,
publication waits and slot-reuse synchronization to BlockPingpong.

| Component | Responsibility |
|---|---|
| `BlockPingpong.cpp` | Match producers and consumers, construct LDS rings and fragment queues, schedule compute and memory phases |
| `ConvertBufferLdsCopies.cpp` | Convert eligible producers without choosing their synchronization policy |
| `ScheduleBufferToLds.cpp` | Implement standalone synchronization and pipeline publication in separate passes; derive waits from slot lifetimes |

Copy conversion is followed by exactly one synchronization owner:
`SynchronizeBufferLdsCopies` for standalone copies or
`ScheduleBufferToLds` for the pipeline. A BlockPingpong request does not
by itself prove a partial wait safe.

## Prefetch distances

| Option | Meaning |
|---|---|
| `block_pingpong_async_global_prefetch` | Number of resident BK generations, 2–8; unspecified uses the BufferLds scheduler default of 4 |
| `block_pingpong_async_local_prefetch` | Number of prefetched A/B register fragments, 1–7 and less than the resolved global distance |

Both launch options are included in compiled-kernel metadata by the common
Triton compiler. The local prefetch distance counts LDS-to-register fragments.

The backend passes these settings explicitly to layout selection, pipeline
preparation and scheduling passes. They are not module attributes. Preparation
records each selected loop's concrete buffer count for pipeline expansion and
static-slot finalization.

The selected tile and ring depth must fit the target LDS budget. Ordinary
VGPR producers retain their existing two-slot policy. Unsupported BufferLds
placement retains the existing copy fallback.

Transport selection does not require a single dot in the module. BlockPingpong
checks each candidate loop independently, including its target LDS capacity.
Resolved prefetch distances are checked before preparation changes the loop;
an invalid default/explicit combination is diagnosed rather than failing
partway through fragment-queue construction.

The scheduler builds independent static LDS slots and an SSA fragment queue.
A publication wait completes the generation required by its consumer while
allowing newer copies to remain pending when proven safe. A control rendezvous
and LDS-read drain protect slots before overwrite. Unknown aliasing or effects
retain conservative waits. Epilogue consumers drain outstanding transfers.

Dynamic reduction bounds remain runtime values throughout the pipeline. No
wrapper hides constant bounds, and no late pass restores them.

## Tests

### Runtime-K wave-group optimization

For eligible K-contiguous A / N-contiguous B tiles, the compiler balances the
scalar A and matrix B fragments by changing the warp grid from `[2,4]` to
`[4,2]`. LEGACY uses software accumulator transpose; TRANSPOSE keeps its
hardware LTS implementation. Other accumulator layouts keep their existing
path. No K bound is required for this layout transformation.

The late scheduling optimization matches four resident generations, one
prefetched fragment, eight waves, independent static LDS slots and four b64
plus four matrix reads per phase. It staggers the two wave groups while
retaining runtime loop bounds and the ordinary fill/drain code. Unsupported
fragment footprints or synchronization patterns retain the ordinary pipeline.
The wide B copies belonging to these slots disable cache swizzle; the narrow
A copies retain their descriptor policy.

Before unrolling, a shortened main loop is normalized to an empty loop when
its upper bound falls below its lower bound. This preserves zero-trip semantics
for short runtime K instead of allowing unsigned unroll arithmetic to create
spurious remainder iterations.

The runtime-K regression executes zero, short, wrap-boundary, 4064, 4096 and
5120 reductions with one binary per dtype/layout, including fp16/bf16 and the
software/hardware accumulator-transpose variants. It is included through
`test_buffer_to_lds_gpu.py` in `regression.sh`.

### Coverage

- `test_buffer_to_lds_passes.py`: options, copy conversion, encoding/lowering
  boundaries, full-pipeline compilation, ordinary layout preservation and
  unsupported producer fallback. These tests do not require a GPU.
- `test_buffer_to_lds_gpu.py`: numerical correctness with runtime K,
  first/wrap/final generations, deeper fragment queues, descriptor addressing,
  batched tails, and ordinary non-async BlockPingpong.
- `test_buffer_publication_wait.py`: partial-wait lifetime proofs and conservative
  barriers when the proof fails, buffer asyncmark code generation, and
  resource-local cache-stride fallback.

The removed static-only strategy selector, NSplitPingPong split-output
transformations and transactional trial compilation are not restored. The
dynamic-K mechanisms above require no new option or static-reduction proof.

A proven publication barrier carries `hcu.block_pingpong.drop_lgkmcnt` in
TTGIR. This suppresses only its implicit `lgkmcnt(0)` wait; the async-copy
wait and `s_barrier` remain, and a later rendezvous drains LDS reads before
the slot can be reused. Unproved publication retains the conservative wait.

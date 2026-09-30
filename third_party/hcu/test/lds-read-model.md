# LDS consumer model: implementation and validation status

This change is **not yet the all-architecture/all-variant delivery**. The
available machine has eight gfx936 devices; other architectures cannot be
hardware-validated here. The local HCU KB is unavailable. Do not describe the
instruction inventory as a table of validated target support.

## Model boundaries

`LdsReadModel.h` separates instruction identity from physical access semantics:

* The ten named matrix intrinsics are inventoried from `IntrinsicsHCU.td`.
  Their dimensions and element widths do not imply their phase or register map.
* A physical read trace contains bit-addressed spans per source lane and phase.
  A lane can contribute several spans or appear in several phases. This can
  express sub-byte elements, gathers, and split transactions.
* Result pieces describe destination lane/register/bit and their source bits.
  Validation checks source coverage, destination overlap, and complete output
  coverage. Source-address lanes are not destination-register lanes.
* Bank geometry and broadcast semantics are explicit. `None` counts different
  lanes requesting the same word separately; `SameBankWord` merges them.
  Neither behavior is inferred from the scalar element type.

The existing wave64 byte-access profiles adapt their physical addresses into
this model before scoring. The legacy matrix selector is shared by placement
eligibility and planning; lowering uses the same intrinsic inventory and tile
dimensions. Existing target and encoding checks remain mandatory. Ordinary
legacy local loads share the vector-width bound with the configuration search: for example,
a 4B copy splits an 8B K fragment into two reads, each using the b32 phase
table. Final LLVM load combining still needs instruction-level validation;
a vector-width bound alone does not prove the final ISA sequence.

## Search objective and proof boundary

The placement search enumerates all power-of-two wave groups, row chunks,
wrap counts, legal wrap steps, and both transfer orders within its existing
row-permutation family. Automatic width selection compares all legal 4/8/16B
copy widths rather than returning the first valid width.

The lexicographic objective is worst phase way, sum of phase ways, global
32-byte sector estimate, copy instruction count, then deterministic locality
tie-breakers. Constraints on alignment, tile resources and contiguous reads
still reject illegal candidates before comparison. A lower bank cost may cost
global throughput; performance baselines must not be silently lowered.

This is optimality within the enumerated placement family and supplied read
profiles, **not all possible LDS layouts or all possible read instructions**.
Generic format/padbyte intrinsics, target availability and alternate consumer
instruction sequences are not yet candidates.

## Tests

`test_buffer_lds_config.py` builds two standalone C++17 tests:

* Existing configuration search regressions check transport bijections and search results.
  Historical GPU fixtures are legal reference placements, not frozen winners.
* The new model test compares scoring against a bit-expanding independent
  oracle; it covers 4/8/16/32/64/128-bit spans, misalignment, gathers, repeated
  lanes, broadcast modes, and invalid addresses. A golden b16 result map checks
  lane 0 receives 0,32,64,96,16,48,80,112. It independently enumerates copy
  widths, wrap, chunks, wave groups and transfer order for a small matrix tile.

The oracle does not call the production bank scorer or physical-address
conversion. Its exhaustive test proves selection for that bounded tile family;
it does not independently establish hardware phase tables.

Retain pure FP8/FP16 copy tests, BF16 schedule/tail tests and the existing opt-in
FP16 performance gates. Pure producer-layout readback alone does not validate
the MMAC consumer mapping. No FP8 GEMM test is required by this change.

## Remaining completion gates

1. Obtain target-specific source-lane, result-register, phase and broadcast
   semantics for m32x64_b4, m32x8_b32, all alt/transpose forms, and generic
   format/padbyte operations. Do not derive these from the b16/b8 tables.
2. Add architecture capability profiles from pinned LLVM instruction predicates
   and hardware specifications, including bank geometry and wrap units.
3. Make ordinary and matrix LDS lowering consume the complete selected read
   configuration; validate the final instruction sequence, including LLVM's ds_read2
   combining and its phase semantics, not just intrinsic names/vector widths.
4. Add real consumer-fragment GPU readback and isolated bank-conflict probes for
   every supported architecture/variant pair. Kernel-average counters cannot
   prove an individual instruction's phase map.
5. Validate all variants on their supported hardware before declaring the
   complete model release-ready. Unknown profiles must not enter optimization.

## Validation on 2026-09-21

On gfx936, with performance and copy correctness isolated on devices 2/3
(at most two cards):

* Standalone configuration search/model tests: 2 passed; the expanded independent oracle was
  rerun after its additional negative and geometry checks and passed.
* Compiler pass/options/layer tests: 143 passed.
* Pure FP8/FP16 transport tests: 120 passed, including 4/8/16B copies, both
  contiguous dimensions and both transfer orders.
* Consumer-fragment tests: 60 passed, including both named matrix instructions,
  both operand orientations and FP8 legacy-emulation/native fragment widths.
  Loaded data stays in consumer encoding through global store; only the output
  coordinates are converted. ISA checks require the expected matrix instruction.
* BF16 schedules and tails: 12 passed.
* Existing opt-in performance gates: 10 passed with unchanged 1% thresholds.

Selected graph-replay results, expressed as FLOP-count/time estimated TFLOPS
(not profiler counters):

| MNK | Layout | Reference | Current |
| --- | --- | ---: | ---: |
| 4096³ | row-col, static K | 288.11 | 287.99 |
| 5120³ | row-col, static K | 338.52 | 338.71 |
| 4096³ | row-row | 298.36 | 298.42 |
| 5120³ | row-row | 380.76 | 381.02 |

Logs are archived under `hygon_tmp/lds-model-evidence/`; generated compiler
artifacts are in `/tmp/triton-lds-model-width-validation` and
`/tmp/triton-lds-model-consumer-v3`. No new hardware bank-counter claim is
made by these tests. In particular, the tests do not close the all-architecture
or unmodelled-instruction completion gates above.

## Instruction-isolated follow-up (2026-09-21)

The follow-up adds reusable `matrixReadResults`, `matrixReadBankPhases`,
`matrixReadTrace`, and `pairedReadTrace` constructors. Routing is keyed by
opcode and target, not inferred from GEMM dtype or contiguous dimension.
Unknown targets and incomplete profiles return no trace. These APIs do not
automatically select a new MMAC consumer instruction in lowering.

### Register routing

`run_lds_instruction_probe.py` compiles one instruction at a time and records
all 64 lanes' four destination VGPRs. Multiple source-index digits distinguish
all elements even for b4. On gfx936, b4, b8, b16, b16-alt, and b32 passed both
full source-element permutation and XOR-basis reconstruction checks. The
production model stores those measured bases separately from bank phases.
Execution checks the actual visible GPU and rejects non-gfx936 hosts.

### Differential bank experiments

`probe_lds_phase_pairs.py` uses `lds_bank_probe.hip`. One selected lane reads
bank words 4..7; another reads either 8..11 (control) or different words in
4..7 (collision). Other lanes broadcast words 0..3. Each case executes 1024
identical matrix-read instructions. The script refuses unexpected nonzero
control counts or non-integral per-instruction counter deltas.

For each opcode, 280 representative-to-remaining-lane pairs reconstruct eight
groups covering all 64 lanes. This validates group membership under this
experiment, not temporal issue order, every possible address pattern, or
exhaustive transitivity across all 2016 lane pairs.

| gfx936 opcode | Group containing lane 0 | Conflict-cycle delta per iteration |
| --- | --- | ---: |
| m32x16_b16 | 0,1,4,5,18,19,22,23 | 1 |
| m32x32_b8 | 0,2,4,6,17,19,21,23 | 1 |
| m32x16_b16_alt | 0,2,4,6,17,19,21,23 | 1 |
| m32x8_b32 | 0,1,2,3,20,21,22,23 | 1 |
| m32x64_b4 | 0,1,2,3,4,5,6,7 | **2** |

All eight groups for the first four opcodes match the corresponding model
tables. b4 routing is available, but its complete bank-cost trace remains
disabled: the different penalty must not be silently interpreted as the b16
service model. A group partition alone is insufficient to establish that cost.

A second b4 experiment used uniform lane strides 16/32/64/128 bytes. Conflict
counts were 0/8192/24576/57344 over 1024 instructions, respectively. Those match
8 groups with ways 1/2/4/8 and **unit** conflict increments, whereas the sparse
pair experiment increments by two. Consequently a blanket `penalty=2` would
also be wrong. Resolving this discrepancy needs a richer request/phase model
or authoritative hardware semantics; no guessed b4 profile is enabled.

### LLVM read2

`run_lds_bank_probe.py` profiles read2_b32, read2_b64 and both ST64 forms.
ST64 scales the instruction offsets by 64 component widths. Each offset has
its own component phases; treating read2_b32 as a contiguous b64 is incorrect.

| Component | Per-lane stride (bytes) | Measured conflict cycles / 1024 iterations | Model way |
| --- | ---: | ---: | ---: |
| b32 | 4 | 0 | 1 |
| b32 | 8 | 4096 | 2 |
| b32 | 128 | 126976 | 32 |
| b64 | 8 | 0 | 1 |
| b64 | 16 | 8192 | 2 |
| b64 | 128 | 122880 | 16 |

Normal and ST64 forms give the same counters for these cases. Ordinary offsets
are 0/32 for b32 and 0/16 for b64; ST64 uses 0/1. LDS initialization adds 256
index-access cycles but no conflicts. Component-wise scoring explains all 12
measurements. CPU tests cover offset units, result packing, invalid offsets,
duplicate/missing lanes, and expected bank costs. This closes the isolated
read2 experiment, **not** general final-ISA sequence matching after arbitrary
LLVM vector fusion; that remains a separate integration gate.

### Cross-compilation and remaining external requirements

| Target | Actual local toolchain result |
| --- | --- |
| gfx936 | First five variants compile and execute; later variants rejected |
| gfx938 | Nine variants compile; m64x16_b8_alt4 reports invalid operand; no matching GPU available |
| gfx926 | Target accepted, tested matrix mnemonics rejected as unsupported |
| gfx92a / gfx946 / gfx948 | Compiler rejects target ID |

Assembler acceptance is not hardware validation. To finish all-target
acceptance requires matching hardware and a compiler/specification supporting
the rejected variants. Generic format/padbyte variants, b4 cost semantics,
and full final-ISA-to-configuration verification are still open, not marked complete.

Reproduction (select an idle GPU first; never occupy more than three):

```bash
HIP_VISIBLE_DEVICES=2 python third_party/hcu/test/run_lds_instruction_probe.py \
  --arch gfx936 --execute --output /tmp/lds-routing-new
HIP_VISIBLE_DEVICES=2 python third_party/hcu/test/run_lds_bank_probe.py \
  --output /tmp/lds-read2-new
```

The profiler runner accepts explicit installation paths; it uses the existing
`/opt/rocm-6.3.3/bin/xprof` and XCU's `status` mode without changing `/opt/rocm`.
All raw captures, per-pair metrics, compiler logs and mappings from this run
are under `hygon_tmp/lds-model-completion/`. The compiler rebuilt successfully;
the standalone tests passed, and all 180 transport/consumer GPU tests passed.

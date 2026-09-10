"""Pure multi-die dispatch requests: no device queries, environment or HIP ABI."""
from dataclasses import dataclass


@dataclass(frozen=True)
class XCDLaunchMetadata:
    """Per-launch XCD dispatch metadata; does not remap tiles inside the kernel.

    Usage: kernel[grid](*args, xcd_metadata=XCDLaunchMetadata(...)).
    Omitting xcd_metadata or passing None uses the original default launch.
    XCDLaunchMetadata() explicitly requests linear dispatch with chunk_size=1;
    it is not the same as omitting the argument. This metadata is not part of
    the kernel compilation cache key.

    Reading the diagrams
    --------------------
    The three 8x4 diagrams use the SAME logical tile layout: four
    XCDs, die_mask=0, one tile per workgroup, and no kernel-side remapping.
    Here tile_id = pid_x + 8 * pid_y; X increases rightward, Y downward.
    Each bordered cell is one tile, NOT one XCD or one dispatch block.

        +------+    04 = logical tile ID
        | 04:C |     C = executing XCD (A=XCD0, B=XCD1, C=XCD2, D=XCD3)
        +------+

    Tile IDs stay in place across diagrams; only the executing XCD changes.
    All sizes below count workgroups, not threads, bytes, or matrix elements.

    linear: consecutive chunks
    --------------------------
    Assign chunks of chunk_size workgroups to enabled XCDs. The runtime/device
    determines traversal and row phase; this API does not specify either.
    Do NOT assume that each logical row always starts with XCD0, or that the
    general rule is xcd = (row_major_tile_id // chunk_size) % num_xcds.
    Example: XCDLaunchMetadata(mode="linear", chunk_size=2).
    The following 8x4 layout matches the tested PMD probe, with start=XCD0:

        +------+------+------+------+------+------+------+------+
        | 00:A | 01:A | 02:B | 03:B | 04:C | 05:C | 06:D | 07:D |
        +------+------+------+------+------+------+------+------+
        | 08:A | 09:A | 10:B | 11:B | 12:C | 13:C | 14:D | 15:D |
        +------+------+------+------+------+------+------+------+
        | 16:A | 17:A | 18:B | 19:B | 20:C | 21:C | 22:D | 23:D |
        +------+------+------+------+------+------+------+------+
        | 24:A | 25:A | 26:B | 27:B | 28:C | 29:C | 30:D | 31:D |
        +------+------+------+------+------+------+------+------+

    Each chunk is two consecutive tiles: [0, 1] -> XCD0, [2, 3] -> XCD1,
    [4, 5] -> XCD2, [6, 7] -> XCD3, [8, 9] -> XCD0, and so on.
    In this example: xcd = (tile_id // 2) % 4. chunk_size=1 instead rotates
    after every workgroup. Flattening order and the physical starting XCD
    shown here are not guarantees for arbitrary runtime environments.

    Row phase is a separate issue from the starting XCD of a launch. The
    original design's Linear chunk=1 drawing shows this 4x5 tile layout:

        +------+------+------+------+
        | 00:A | 01:B | 02:C | 03:D |   row 0: A B C D
        +------+------+------+------+
        | 04:D | 05:A | 06:B | 07:C |   row 1: D A B C
        +------+------+------+------+
        | 08:C | 09:D | 10:A | 11:B |   row 2: C D A B
        +------+------+------+------+
        | 12:B | 13:C | 14:D | 15:A |   row 3: B C D A
        +------+------+------+------+
        | 16:A | 17:B | 18:C | 19:D |   row 4: A B C D
        +------+------+------+------+

    Here the row starts shift WITHIN ONE grid. This reproduces the design
    drawing, not an additional PMD measurement or a universal row-shift rule.
    A single per-launch starting offset cannot explain this row variation.
    Separately, successive PMD launches can start on different XCDs. Neither
    case is controlled by chunk_size. Validate the actual dispatch mapping
    before pairing Linear with a kernel-side PID remap.

    fixed: fixed spatial partitions
    -------------------------------
    Example: XCDLaunchMetadata(mode="fixed").
    The current PMD 8x4 probe on four XCDs produces four quadrants, each a
    contiguous 4x2 tile region:

        +------+------+------+------+------+------+------+------+
        | 00:A | 01:A | 02:A | 03:A | 04:B | 05:B | 06:B | 07:B |
        +------+------+------+------+------+------+------+------+
        | 08:A | 09:A | 10:A | 11:A | 12:B | 13:B | 14:B | 15:B |
        +------+------+------+------+------+------+------+------+
        | 16:C | 17:C | 18:C | 19:C | 20:D | 21:D | 22:D | 23:D |
        +------+------+------+------+------+------+------+------+
        | 24:C | 25:C | 26:C | 27:C | 28:D | 29:D | 30:D | 31:D |
        +------+------+------+------+------+------+------+------+

    XCD0 owns the upper-left region; XCD1 upper-right; XCD2 lower-left;
    XCD3 lower-right. Fixed does NOT mean pinning the entire kernel to one
    XCD. The dispatch rule determines partition sizes, not block_x/block_y.
    Keep all mask/chunk/block fields at their defaults in this mode.
    Do not extrapolate this observed layout to other shapes or topologies.

    block: rectangular dispatch blocks
    ----------------------------------
    Partition the workgroup grid into block_x by block_y rectangles;
    all workgroups in a rectangle execute on the same XCD.
    Block size does NOT fix the first XCD or imply an A B C D sequence on
    every block row. Distinguish row/column-dependent assignment WITHIN a
    grid from a starting-XCD change BETWEEN launches.
    Example: XCDLaunchMetadata(mode="block", block_x=2, block_y=2).
    The current PMD shifts the XCD sequence between adjacent block rows.
    Normalizing the first block's XCD to XCD0 gives:

        +------+------+------+------+------+------+------+------+
        | 00:A | 01:A | 02:B | 03:B | 04:C | 05:C | 06:D | 07:D |
        +------+------+------+------+------+------+------+------+
        | 08:A | 09:A | 10:B | 11:B | 12:C | 13:C | 14:D | 15:D |
        +------+------+------+------+------+------+------+------+
        | 16:B | 17:B | 18:C | 19:C | 20:D | 21:D | 22:A | 23:A |
        +------+------+------+------+------+------+------+------+
        | 24:B | 25:B | 26:C | 27:C | 28:D | 29:D | 30:A | 31:A |
        +------+------+------+------+------+------+------+------+

    Read the blocks, not just the first tile row:
      - Block row 0 (tile rows 0/1): A B C D.
      - Block row 1 (tile rows 2/3): B C D A, NOT A B C D again.
    The first 2x2 block is tiles [0, 1; 8, 9] -> XCD0. The block directly
    below it is [16, 17; 24, 25] -> XCD1, not XCD0.
    In this example: xcd = (pid_x // 2 + pid_y // 2) % 4. This is NOT a
    simple round-robin over row-major flattened 2x2 blocks, and differs
    from Fixed's four large partitions.

    The design also describes special 1/2/3-column grouping with different
    starting dies per group. The current HIP config has no separate selector
    for those modes; block_x/block_y must not be presented as that selector.
    Neither this 2x2 example nor its formula specifies all masks, grid shapes,
    edge blocks, or column-grouping variants. Confirm their actual mapping
    separately before using it for kernel-side remapping.

    Model observations, not hardware guarantees
    ------------------------------------------
    Linear/Block may start on different XCDs across successive launches.
    For a starting XCD s in the tested 8x4 layouts, add s to the example XCD
    IDs modulo 4; this preserves, rather than replaces, the row offsets.
    This API does not select that starting XCD or guarantee execution order.
    The diagrams are design/model examples as labeled, not validation on real
    hardware. Tests use the model's tested domain (even 2D Fixed grids,
    divisible 2D Block grids). This object does not query devices or impose
    those model-specific restrictions as general hardware limits.
    """

    # Dispatch policy: consecutive chunks, fixed partitions, or rectangles.
    mode: str = "linear"
    # Linear/Block only. Special value 0 allows ALL available XCDs to execute.
    # Otherwise bit 0/1/2/3 enables XCD0/1/2/3; e.g. 0b0101=5 enables XCD0/2.
    # This selects eligible XCDs; it does not replicate the kernel on each XCD.
    # The current model ignores partial-mask filtering; we still pass the mask
    # through unchanged. Fixed requires the default value 0.
    die_mask: int = 0
    # Linear only: positive number of consecutive workgroups per chunk.
    # Not num_warps or compiler pipeline stages. Block/Fixed require default 1.
    chunk_size: int = 1
    # Block only: positive dispatch-block extents along workgroup grid X/Y.
    # A 2x2 block covers four workgroups; not HIP blockDim or GEMM BM/BN.
    # Linear/Fixed require both fields to remain at their default value 1.
    block_x: int = 1
    block_y: int = 1

    def __post_init__(self):
        if self.mode not in ("linear", "fixed", "block"):
            raise ValueError("XCD mode must be linear, block or fixed")
        for name in ("die_mask", "chunk_size", "block_x", "block_y"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an int, not bool")
        if self.die_mask < 0:
            raise ValueError("die_mask must be nonnegative (0 means all XCDs)")
        if self.chunk_size < 1 or self.block_x < 1 or self.block_y < 1:
            raise ValueError("chunk and block dimensions must be positive")
        if self.mode != "block" and (self.block_x != 1 or self.block_y != 1):
            raise ValueError("block_x/block_y only apply to block mode")
        if self.mode == "block" and self.chunk_size != 1:
            raise ValueError("chunk_size only applies to linear mode")
        if self.mode == "fixed" and (self.die_mask != 0 or self.chunk_size != 1):
            raise ValueError("fixed does not accept mask/chunk overrides")

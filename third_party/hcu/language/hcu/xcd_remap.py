"""Choose which logical tile each program computes; do not change its launch.

Input pid is a scalar, flattened raw program ID. The returned ID selects a
tile in the caller's data layout. These helpers do not read the hardware XCD ID,
query topology, move workgroups, or set launch metadata.

Each function documents its own mapping, 2D tile diagrams, and kernel usage.
For an XCD-locality benefit, actual dispatch must match raw_pid % NUM_XCDS
grouping (a consistent rotation of physical owners is fine). Arbitrary
Linear/Fixed/Block settings, 2D launch grids, or masks need not satisfy this
condition. The diagrams are logical mappings, not proof of XCD placement
or L2 hits.
"""

import triton
import triton.language as tl

__all__ = ["remap_xcd", "remap_xcd_chunked", "remap_xcd_fixed"]


@triton.jit
def remap_xcd_chunked(
    pid, GRID_MN, NUM_XCDS: tl.constexpr = 2, CHUNK_SIZE: tl.constexpr = 2
):
    """Give each modulo group CHUNK_SIZE adjacent tiles per complete window.

    Requires 0 <= pid < GRID_MN and positive NUM_XCDS/CHUNK_SIZE. A final
    incomplete window is unchanged. Pass NUM_XCDS explicitly for the target
    dispatch rather than assuming the default of two matches the device.

    Example: 4x4 logical tiles, GRID_MN=16, NUM_XCDS=4, CHUNK_SIZE=2.
    Rows are tile_m, columns tile_n; each cell stays at tile_m * 4 + tile_n
    and shows tile_id:group. A/B/C/D are raw_pid % 4 groups, not measured
    XCD IDs. Positions below are logical tiles, NOT raw PID positions.

    No remap: tile_id = raw_pid
               tile_n ->
           +------+------+------+------+
    m = 0  |  0:A |  1:B |  2:C |  3:D |
    m = 1  |  4:A |  5:B |  6:C |  7:D |
    m = 2  |  8:A |  9:B | 10:C | 11:D |
    m = 3  | 12:A | 13:B | 14:C | 15:D |
           +------+------+------+------+

    Chunked remap: each group gets 2 adjacent tiles per 8-tile window
           +------+------+------+------+
    m = 0  |  0:A |  1:A |  2:B |  3:B |  window 0
    m = 1  |  4:C |  5:C |  6:D |  7:D |
           +------+------+------+------+
    m = 2  |  8:A |  9:A | 10:B | 11:B |  window 1
    m = 3  | 12:C | 13:C | 14:D | 15:D |
           +------+------+------+------+

    Group A's raw PIDs [0,4,8,12] now compute tiles [0,1,8,9].
    CHUNK_SIZE counts logical tiles per group per software window; it is
    independent of the hardware launch chunk_size. Windows are linear
    ranges and need not align with matrix rows for other shapes.

    Inside a kernel with grid=(NUM_TILES_N, NUM_TILES_M):
        raw_pid = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
        total = NUM_TILES_M * NUM_TILES_N
        tile_id = remap_xcd_chunked(raw_pid, total, NUM_XCDS=4, CHUNK_SIZE=2)
        tile_m, tile_n = tile_id // NUM_TILES_N, tile_id % NUM_TILES_N
        # Use tile_m/tile_n, not the original program IDs, for addresses.
    """
    # Group under the round-robin assumption, not a hardware XCD ID read.
    xcd = pid % NUM_XCDS
    # Leave the incomplete final window in its original order. At the exact
    # boundary the formula below also returns pid, so '>' is intentional.
    if pid > (GRID_MN // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    # Position among the programs in this modulo group.
    local_pid = pid // NUM_XCDS
    # Split that position into a software window and an offset within it.
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    # Window base + this group's tile range + offset inside that range.
    new_pid = chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk
    return new_pid


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 2):
    """Give each modulo group a contiguous tile range over the whole grid.

    Requires 0 <= pid < GRID_MN and positive NUM_XCDS. Uneven group sizes
    are supported, including GRID_MN < NUM_XCDS. NUM_XCDS is supplied by
    the caller, not queried from the device or inferred from launch metadata.

    Example: 4x4 logical tiles, GRID_MN=16, NUM_XCDS=4.
    Rows are tile_m, columns tile_n; each cell stays at tile_m * 4 + tile_n
    and shows tile_id:group. A/B/C/D are raw_pid % 4 groups, not measured
    XCD IDs. Positions below are logical tiles, NOT raw PID positions.

    No remap: tile_id = raw_pid
               tile_n ->
           +------+------+------+------+
    m = 0  |  0:A |  1:B |  2:C |  3:D |
    m = 1  |  4:A |  5:B |  6:C |  7:D |
    m = 2  |  8:A |  9:B | 10:C | 11:D |
    m = 3  | 12:A | 13:B | 14:C | 15:D |
           +------+------+------+------+

    Full-grid remap: each group gets one contiguous range over all 16 tiles
           +------+------+------+------+
    m = 0  |  0:A |  1:A |  2:A |  3:A |
    m = 1  |  4:B |  5:B |  6:B |  7:B |
    m = 2  |  8:C |  9:C | 10:C | 11:C |
    m = 3  | 12:D | 13:D | 14:D | 15:D |
           +------+------+------+------+

    Group A's raw PIDs [0,4,8,12] now compute tiles [0,1,2,3]. The ranges
    happen to fill whole rows here; other shapes can cross row boundaries.
    This helper uses the total tile count, not the matrix dimensions.

    Inside a kernel with grid=(NUM_TILES_N, NUM_TILES_M):
        raw_pid = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
        total = NUM_TILES_M * NUM_TILES_N
        tile_id = remap_xcd(raw_pid, total, NUM_XCDS=4)
        tile_m, tile_n = tile_id // NUM_TILES_N, tile_id % NUM_TILES_N
        # Use tile_m/tile_n, not the original program IDs, for addresses.
    """
    # Size of the larger groups when the grid does not divide evenly.
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # The first remainder groups get one extra tile (the "tall" groups).
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        # Even division: every group has pids_per_xcd tiles.
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
    # Assumed owner group and the program's position within that group.
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Place the larger groups first, followed by the smaller groups.
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def remap_xcd_fixed(pid, GRID_M, GRID_N, NUM_XCDS: tl.constexpr = 2):
    """Map round-robin program groups to fixed rectangular tile regions.

    This is a software spatial mapping, NOT a helper to apply on top of
    Fixed launch. For 2/4 groups, it requires actual dispatch compatible with
    raw_pid % NUM_XCDS grouping, e.g. a verified Linear chunk=1
    layout. A consistent rotation of physical XCD owners is allowed.

    pid: scalar flattened raw program ID, 0 <= pid < GRID_M * GRID_N.
    GRID_M / GRID_N: positive logical tile rows / columns; runtime scalars
    are supported, as are compile-time constants. Positivity is a caller
    precondition, like the valid pid range, not a compile-time restriction.
    NUM_XCDS: defaults to 2; supplied by the caller, not queried from the device.
    Only 2 and 4 enable remapping; 1 and all other values return pid unchanged.
    This fallback only skips the optimization; it does not validate topology.
    Returns: row-major tile ID; decode with // GRID_N and % GRID_N.

    Partition convention (M increases downward, N increases rightward):

        NUM_XCDS=1      NUM_XCDS=2      NUM_XCDS=4
        +---------+     +----+----+     +----+----+
        |         |     |    |    |     | A  | B  |
        |    A    |     | A  | B  |     +----+----+
        |         |     |    |    |     | C  | D  |
        +---------+     +----+----+     +----+----+
        identity        split N        split M and N

    Arbitrary positive tile dimensions are supported: for GEMM, pass
    ceil(M / BLOCK_M) and ceil(N / BLOCK_N), without padding the launch grid.
    With 1 group pid is unchanged. With 2/4 groups, the largest divisible
    top-left rectangle is partitioned as above. Remaining raw PIDs cover
    the right strip first, then the bottom rows, both in row-major order.
    Edges need not remain with the neighboring region's owner. If there is
    no complete rectangle (e.g. one tile row with 4 groups), return pid.
    Every tile is visited once; the kernel's normal element masks still
    handle partially filled tiles. No extra workgroups are required.

    Example: GRID_M=3, GRID_N=5, NUM_XCDS=4. T marks edge tiles, not an XCD:

        +----+----+----+----+----+
        | A  | A  | B  | B  | T  |
        | C  | C  | D  | D  | T  |
        +----+----+----+----+----+
        | T  | T  | T  | T  | T  |
        +----+----+----+----+----+

    Raw PIDs 0..7 cover the 2x4 interior; 8..14 map to edge tile IDs
    [4, 9, 10, 11, 12, 13, 14]. Edge ownership still follows raw_pid % 4;
    the final owner regions are not necessarily perfect rectangles.
    The 2-group left/right split is this helper's software convention,
    not a claim about an unverified two-XCD hardware Fixed launch.

    Example: GRID_M=4, GRID_N=8, NUM_XCDS=4. Each cell stays at its logical (m, n)
    position and shows tile_id:group. A/B/C/D are raw_pid % 4 groups,
    not hardware IDs read by this helper:

          +------+------+------+------+------+------+------+------+
    m=0   | 00:A | 01:A | 02:A | 03:A | 04:B | 05:B | 06:B | 07:B |
    m=1   | 08:A | 09:A | 10:A | 11:A | 12:B | 13:B | 14:B | 15:B |
          +------+------+------+------+------+------+------+------+
    m=2   | 16:C | 17:C | 18:C | 19:C | 20:D | 21:D | 22:D | 23:D |
    m=3   | 24:C | 25:C | 26:C | 27:C | 28:D | 29:D | 30:D | 31:D |
          +------+------+------+------+------+------+------+------+

    Group A's raw PIDs [0,4,8,12,16,20,24,28] now compute tiles
    [0,1,2,3,8,9,10,11]: one 2x4 region, rather than one linear strip.

    Inside a kernel with grid=(GRID_N, GRID_M):
        raw_pid = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
        tile_id = remap_xcd_fixed(raw_pid, GRID_M, GRID_N, NUM_XCDS=4)
        tile_m, tile_n = tile_id // GRID_N, tile_id % GRID_N
        # Use tile_m/tile_n for addresses; launch with verified round-robin
        # dispatch, NOT xcd_metadata=XCDLaunchMetadata(mode="fixed").

    These are helper constraints, not additional restrictions on Fixed launch.
    This function does not promise execution order or a performance gain.
    """
    if NUM_XCDS != 2 and NUM_XCDS != 4:
        return pid
    parts_n: tl.constexpr = 2
    parts_m: tl.constexpr = NUM_XCDS // parts_n
    region_m = GRID_M // parts_m
    region_n = GRID_N // parts_n
    if (region_m == 0) | (region_n == 0):
        return pid

    core_m = region_m * parts_m
    core_n = region_n * parts_n
    core_size = core_m * core_n
    if pid >= core_size:
        # Splitting N in two leaves at most one right-edge column.
        right_tile = (pid - core_size) * GRID_N + core_n
        # Right-edge PIDs precede core_m * GRID_N; bottom rows keep their PID.
        # With even GRID_N the right-edge interval is empty.
        return tl.where(pid < core_m * GRID_N, right_tile, pid)

    # Select the region from the assumed owner, not from the raw 2D position.
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Traverse the region row-major, then offset into the full tile matrix.
    tile_m = (xcd // parts_n) * region_m + local_pid // region_n
    tile_n = (xcd % parts_n) * region_n + local_pid % region_n
    return tile_m * GRID_N + tile_n

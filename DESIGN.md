# Design: Warp-Specialized Mojo GPU Chemistry Kernel

Status: **Hopper M4 spill-free implementation; grid-1 correctness established**
Date: 2026-07-16
Authors: Ben Wibking + design review sessions

This document is the complete, self-contained plan for rewriting the
primordial-chemistry reproducer (`reproducer.mojo` / `reproducer.cpp`) as a
**Singe-style warp-specialized GPU kernel in Mojo**, following Modular's
**structured kernel pattern** (TileIO / TilePipeline / TileOp). It is written
so a fresh session on the Linux+GPU machine can execute the plan without any
other context.

---

## 1. Goal and non-goals

**Goal.** Replace the current thread-per-cell GPU integration with a
warp-specialized kernel in which `NUM_WARPS` warps cooperate on a tile of
cells. The structured kernel should eliminate compiler-reported spills if the
toolchain permits it, and in all cases must reduce private-memory traffic and
beat the data-parallel Mojo GPU baseline before it becomes the default. The CPU
and existing C++ GPU entry points remain intact.

**Numerics policy (decided).** *Tolerance-PASS* using the existing packed-file
comparison schema/tolerances and a newly generated grid-wide-policy CPU
reference is acceptable. Bitwise identity with the CPU reference is **not**
required. Error-norm and EOS reductions may reassociate.

**Non-goals (v1).**

- No changes to the chemistry RHS/Jacobian, EOS, ROS2S coefficients,
  tolerances, or step-controller formulas. The grid-wide outer timestep and
  CTA-wide coupling of error estimates are intentional driver/controller
  changes.
- No CUDA/HIP rewrite (the C++ paths stay as-is; Mojo GPU is the new path).
- No persistent-megakernel driver. V1 uses one short grid-timestep prepass and
  one chemistry kernel launch per global collapse step, with a host reduction
  between them.
- No producer/consumer overlap (v2); v1 uses full-CTA phase barriers.

### 1.1 Acceptance gates

The design is complete only when all four gates have evidence from the target
machines:

1. **Correctness**: T0--T8 pass; the GPU output passes the packed final-state
   comparator against the new grid-wide CPU policy reference; partial tiles,
   stopped cells, singular retries, and no-active-cell termination are covered.
2. **Resources**: the sm_90 build reports static and dynamic shared memory,
   registers, spills, stack/local bytes, and launch occupancy. The v1 layout
   uses Hopper's large-shared-memory opt-in and stays below the 227 KiB
   per-block limit. Spills are measured explicitly and must be zero.
3. **Performance**: on Hopper, M4 grid-64 wall time is lower than M1 under the
   same driver policy, inputs, and measurement protocol. Otherwise M1 remains
   the default.
4. **Execution safety**: full and partial 32-cell tiles have no out-of-bounds
   or barrier-divergence failures. M4 is intentionally sm_90-only; gfx90a and
   other 64-lane targets are not acceptance targets.

## 2. Background

### 2.1 Current implementation

- `reproducer.cpp` (~8.5k lines): standalone C++20 TU with CPU/CUDA/HIP
  builds. GPU kernel `advance_collapse_grid_kernel` (line ~7671) is
  thread-per-cell, 128 threads/block; each thread runs a full adaptive ROS2S
  integration for one cell per launch. Host loops over global collapse steps
  (≤1000), one launch per step.
- `reproducer.mojo` (~4.3k lines): CPU-only Mojo port of the same code.
  Key functions:
  - `rhs_specie` (line 257): species rates, 119 CSE temporaries → 14 outputs
  - `rhs_eint` (line 442): energy rate, 162 CSE temporaries → 1 output
  - `jac_nuc` (line 1019): analytic Jacobian, **1,394 CSE temporaries**,
    2,776 lines → 225 outputs (15×15)
  - `integrate_ros2s` (line 3926): ROS2S Rosenbrock integrator
    (3-stage, 2nd-order, embedded error control; γ = 1 − 1/√2 ≈ 0.2928932;
    coefficients at `reproducer.cpp:7118`)
  - `burn_ros2s` (line 4064), `make_collapse_state` (line 4109)
- Physics parameters: `NumSpec = 14`, `Neqs = 15`, redshift 30, initial T=100 K,
  `MaxCollapseSteps = 1000`, perturbation every 20 steps (amplitude 0.2).

### 2.2 Why warp specialization (Singe, PPoPP'14)

The kernel exhibits Singe's three target properties:

1. **Large working set**: per cell, the ROS2S state is ~550 doubles
   (y/ynew/k1/k2/work/dy = 90, J+E = 450, ipvt). The ROCm results table in
   `README.md` shows `vgpr_spill_count` ≈ 2,082–2,375 and
   `private_segment_fixed_size` ≈ 7.3 KB/thread on gfx90a — i.e. the working
   set is 4–6× the register file, and ROCm 6.0/6.1 fail outright.
2. **Irregular computation**: distinct phases (Jacobian DAG eval, LU
   factorization, 3× RHS+solve, step control) with very different
   characteristics; the rate code is full of data-dependent conditionals.
3. **Irregular data access**: machine-generated CSE DAGs with complex
   sharing patterns.

Singe's chemistry kernel (8.5 KB spills/thread → 44 bytes) is the direct
analog: partition the big DAG across warps, keep slices register-resident,
exchange through shared memory, and assign the small sequential
sub-computation (there: QSSA; here: 15×15 LU) to dedicated owner warps.

### 2.3 Why structured Mojo kernels

From Modular's blog series ("Structured Mojo Kernels" parts 1–2):

- **TileIO** owns data movement between memory levels.
- **TilePipeline** owns coordination (barriers, producer-consumer sync),
  platform-abstracted via a `SyncStrategy` trait: hardware `mbarrier` on
  NVIDIA, shared-memory atomic counters + spin-wait (`s_sleep 0` yield) on
  AMD. Only the first thread of each warp increments counters.
- **TileOp** owns compute, incl. register/accumulator lifecycle.
- `TileTensor` is the common currency; configuration and component composition
  are comptime-selected, while runtime `warp_id` branches select roles
  uniformly within each warp. Context managers
  (`with producer.acquire() as tiles:`) pair acquire/release transitions;
  full-CTA barrier uniformity still requires an explicit audit.

We adopt the pattern (these are pattern names, not library APIs) and the
practices: trait-based sync, context-manager acquire/release, comptime
configuration as the autotuning surface, type-enforced component boundaries.

## 3. Environment and constraints

- Mojo via pixi: `mojo >= 1.0.0b2, <2` (currently 1.0.0b2). Build/run:
  `pixi run mojo build|run ...`. Platforms: `linux-64`, `osx-arm64`.
- **The dev Mac cannot run GPU code**: Apple GPU (Metal) has no Float64
  support; a minimal Float64 kernel fails with
  `LLVM ERROR: Failed to verify LLVM IR for Metal`. All GPU compile+test
  steps must run on the Linux+GPU machine. CPU unit tests run anywhere.
- GPU target: NVIDIA Hopper sm_90 with 32-lane warps. M4 deliberately uses
  Hopper's large shared-memory allocation and is not constrained by gfx90a
  LDS. `validate_config()` rejects `WARP_SIZE != 32`.
- Mojo GPU essentials (from the mojo-gpu-fundamentals skill):
  kernels are plain `def` (no decorator), launched with
  `ctx.enqueue_function[kernel](args, grid_dim=..., block_dim=...)`;
  comptime-parameterized kernels must be bound first
  (`comptime k = kernel[params]; ctx.enqueue_function[k](...)`);
  `barrier()` = CTA barrier (`std.gpu`); warp shuffles in
  `std.gpu.primitives.warp`; shared memory via
  `stack_allocation[dtype, address_space=AddressSpace.SHARED](layout)`
  (layout package) or raw arrays via `std.memory.stack_allocation`;
  atomics via `std.atomic.Atomic`.
  **Verify on the GPU machine** whether `std.gpu.sync` exposes named
  barriers / mbarrier in 1.0.0b2; v1 does not need them.
- Testing: `mojo test` subcommand is removed; tests are plain executables
  using `TestSuite.discover_tests[__functions_in_module()]().run()`
  (`std.testing`), run with `pixi run mojo run <test_file>.mojo` (CPU) or
  compiled+run on the GPU machine.

## 4. Target architecture

### 4.1 Execution model

- **V1 configuration**: `DAG_WARPS = OWNER_WARPS = 8`, `TILE_CELLS = 32`,
  `BATCH_CELLS = 32`, `BATCHES_PER_TILE = 1`, `LU_GROUP_WIDTH = 8`, and
  `LU_GROUPS_PER_WARP = 4`. Launch with
  `grid_dim = ceil(num_cells / TILE_CELLS)` and `block_dim = 256`. Compile-time
  checks require `WARP_SIZE == 32`, four LU groups per warp, 32 logical owner
  groups, and a total shared-memory budget below Hopper's 227 KiB limit.
- **Device state layout**: use field-major structure-of-arrays storage, not a
  raw `CollapseState` or one untyped tensor. The logical `DeviceCells` bundle
  contains `Float64` state fields (`rho`, `T`, `e`, `xn[14]`, `time`,
  `density_driver`), an `Int32 completed_steps` buffer, and seven `UInt64`
  statistics fields. Adjacent lanes therefore load adjacent cells for each
  field, and integer metadata is never round-tripped through `Float64`. The
  packed 172-byte C++ record is a host serialization format, not a device
  layout. M1 determines whether Mojo can pass the logical bundle directly or
  whether kernels take its tensors as separate arguments.
- **Single-batch mapping**: every generated DAG slice runs across lanes 0--31,
  so one warp evaluates that slice for all 32 tile cells. During LU/solves,
  physical warp *w* is divided into four independent 8-lane groups. Group *g*
  owns cell `4*w + g`; local lane *l* owns columns/components `l` and `l+8`
  when they exist. All 32 systems factor and solve concurrently. The full
  32x15x15 Jacobian/LU arena remains in Hopper shared memory through k1--k3.
- **Grid-wide physical timestep (decided 2026-07-16)**: every participating
  cell in the grid advances through the same outer interval `dt_grid` during a
  global collapse step.
  - A preparation kernel applies the scheduled perturbation exactly once,
    computes each participating cell's free-fall proposal
    `dt_candidate = tff_reduc * tff`, excludes already-stopped cells and cells
    with `dt_candidate < 10`, and writes one `(minimum, global_cell_id)` pair
    per chemistry tile (`(+inf, INT_MAX)` when none participates). Bounds-only
    lanes are neutral. A non-finite/non-positive density, `tff`, or candidate
    is a failure, not a silently excluded cell. A cell stopped by the local
    `dt_candidate < 10` condition receives the same terminal density-driver
    update as the existing driver before it is frozen.
  - The host copies the per-CTA pairs, performs a stable lexicographic
    reduction by `(dt_candidate, global_cell_id)`, and sets
    `dt_grid = min(dt_candidate)` over the whole active grid. It skips the
    chemistry launch when no finite candidate remains. For a 64^3 grid and
    32-cell tiles this is 8,192 minima (64 KiB of `Float64`) plus 32 KiB of
    cell IDs per outer step. A device reduction may replace it later without
    changing the numerical policy or tie-breaking. The host checks the
    preparation failure flag before reducing or launching chemistry.
    Both device and host use the same comparator:
    `(a.dt < b.dt) or (a.dt == b.dt and a.cell < b.cell)` after finite-value
    validation.
  - In the chemistry kernel, each participating cell recomputes `tff` from the
    prepared state and updates
    `density_driver += dt_grid * density_driver / tff`. Thus the cell that
    sets the minimum advances its driver by at most `tff_reduc`, and every
    other cell advances by a smaller fraction of its local free-fall time.
    Cells whose common update crosses the existing density-driver stop limit
    are marked stopped before the burn and excluded from the controllers.
    The preparation-pass active set defines `dt_grid` for the current step;
    the minimum is not recomputed if one of those cells then crosses the
    density-driver limit during setup. That cell is absent from the next
    preparation pass.
  - There is no persistent stopped flag and no per-cell timestep buffer. Setup
    recomputes the local `dt_candidate < 10` predicate. Such cells are neutral
    for chemistry because the preparation kernel already performed their one
    terminal driver update; setup must not update them a second time. A cell
    that stops on the density-driver limit receives its common driver update
    in setup but, matching the existing driver, no density rescale or burn.
    Both kernels call the same device helper for density, `tff`, and
    `dt_candidate`, preventing a threshold-rounding mismatch between passes.
  - The host computes `next_grid_time = grid_time + dt_grid` once. Every cell
    that is actually integrated commits `collapse.time = next_grid_time`
    directly, rather than accumulating per-cell sums. Consequently all cells
    participating in a completed global step end at the same representable
    physical time. Previously stopped cells remain frozen and do not rejoin.
    The host assigns `grid_time = next_grid_time` and increments
    `completed_global_steps` exactly once only after the chemistry launch
    reports no failure and at least one integrated cell. Every
    participating cell must enter preparation with
    `completed_steps == completed_global_steps` and
    `collapse.time == grid_time` exactly; a mismatch is an invariant failure.
    If all remaining cells stop during setup, the host terminates without
    advancing `grid_time`.
- **Unified CTA controller (decided 2026-07-16)**: each CTA advances its
  participating cells from `x = 0` to `tout = dt_grid` with a **single ROS2S
  step controller and a single physical substep `h`**. There is no per-cell
  horizon or controller state. A validity/participation mask is still required
  for partial CTAs and stopped cells, but it never changes the barrier path.
  Every substep iteration executes one ROS2S trial step
  (J → LU → k₁ → k₂ → k₃ → error) for all participating lanes.
  - Step acceptance: `err_tile = max over participating cells of err_cell`;
    accept iff `err_tile ≤ 1`, else reject and scale the common `h` down. The
    Gustafsson controller (`hacc`/`erracc`) operates on `err_tile`.
  - Final-step truncation is common to the CTA:
    `h = min(h, dt_grid - x)`. On acceptance every participating cell commits;
    an accepted final step assigns `x = dt_grid` exactly, while other accepted
    steps use `x += h`. This avoids lane- or cell-dependent horizon tests.
  - LU-singular retry, `DT_UNDERFLOW`, `TOO_MANY_STEPS` become tile-level
    conditions with the same semantics (fail on the fifth singular
    factorization, so at most four retry halvings; unchanged `max_steps`).
  - Saved vs per-cell control: per-cell `h, hacc, erracc, hopt, x, tout,
    reject, nsing, n_step, n_accept` (~12 scalars/cell, ≈4 KB shared plus
    register traffic) collapse to one tile-level shared set, cached or
    broadcast uniformly as needed.
    Participating cells within a CTA share step counts; different CTAs may
    choose different ROS2S substep sequences while still reaching the same
    grid-wide outer time.
  - Numerical effect: the grid-wide outer timestep and maximum-error CTA
    controller change trajectories relative to independent per-cell
    adaptivity. No monotonic accuracy claim is made; final states must satisfy
    the comparison harness tolerances.
- The grid-wide reduction applies only to the outer collapse interval.
  Synchronizing every ROS2S trial across CTAs would require a grid-wide
  reduction inside each adaptive substep and is explicitly out of scope.
- All threads execute the same phase/barrier sequence. Validity and
  participation masks guard arithmetic and memory accesses, never barriers.
- **Cell ownership**: physical warp *w* owns cells `{4*w, ..., 4*w+3}` through
  four logical 8-lane groups. Every group remains active throughout LU, three
  solves, state derivation, and candidate/error production. Candidate state is
  staged until the CTA-level decision; rejected or singular trials never
  overwrite base `y`.

### 4.2 Driver and phase schedule (one global collapse step)

Before the chemistry kernel:

The host clears `failure` and `integrated_count` for the step. The preparation
kernel overwrites every per-CTA pair slot, including empty tiles, so no value
from a previous global step can participate in the reduction.

0. **Grid-timestep preparation kernel**: apply the deterministic perturbation
   once for this global step; honor `completed_steps`/valid-cell gating;
   compute `dt_candidate`; apply the terminal update and exclude local
   `dt_candidate < 10` stops; reduce remaining candidates to one
   `(minimum, lowest_global_cell_id)` pair per chemistry tile. Report invalid
   physics through the failure buffer. Launch exactly one preparation CTA per
   logical chemistry tile with `block_dim = WARP_SIZE`; lanes 0..31 map to the
   tile's cells and any wider-wave lanes are neutral.
1. **Host reduction**: copy the per-CTA pairs, compute the stable global pair,
   and stop the global loop if all entries are `+inf`. Pass the finite
   `grid_time`, `next_grid_time = grid_time + dt_grid`, and `dt_grid` to every
   CTA in the chemistry launch. Abort before launch if preparation reported
   invalid physics, `dt_grid <= 0`, or `next_grid_time` is non-finite.

Inside the chemistry kernel:

0. **Load**: `CellStateIO` loads the tile's cell states (ρ, T, e, xn[14],
   time, density_driver, completed_steps, stats) global→shared.
1. **Driver setup** (owner warps, per owned cell): recompute `tff` from the
   prepared state and first exclude local-`dt<10` cells without touching their
   already-updated driver. For the remaining cells, update the density driver
   with common `dt_grid`; cells crossing the density-driver limit stop before
   density rescale/burn. For actual participants, apply the density rescale and
   build integrator state y[15] (number densities + e). Reset CTA substep time
   `x = 0`, set `tout = h = dt_grid`, and reset controller history
   (`hacc`/`erracc`). The perturbation is not repeated here.
2. **Substep loop** (until the CTA controller reaches `dt_grid` or fails):
   reset the shared singular flag and candidate-error slots, then execute one
   complete 32-cell trial. All early-exit decisions are CTA-uniform.
   - **P0 base state**: DAG lanes derive floored X[14] and T from immutable
     base `y` for all 32 cells.
   - **P1 Jacobian + LU**: each of eight warps evaluates one generated J slice
     across lanes 0--31 and fills the 32x225 dynamic shared arena. The same
     arena is transformed in place to `E = (1/(h*gamma))I - J`. Four 8-lane
     groups per warp factor 32 systems concurrently, each lane owning up to
     two shared columns. A CTA reduction publishes `tile_singular`; on a
     singular trial the controller halves `h`, discards every candidate, and
     retries the whole tile. The fifth singular factorization fails.
   - **P2 k1**: generated RHS slices fill the 32x15 ydot stage; the 32 groups
     solve with the retained shared LU factors and form the first stage state
     and `k2` seed.
   - **P3 k2**: generated RHS slices evaluate the first stage; groups solve and
     form the second stage state plus `work`.
   - **P4 k3 + candidate**: generated RHS slices evaluate the second stage;
     groups solve k3, form each candidate and embedded error, and write one
     scalar error per participating cell. The shared LU arena is then dead.
3. **CTA decision**: after the 32-cell trial completes, reduce
   `err_tile = max(err_cell)` over participating cells. Apply the unchanged
   ROS2S/Gustafsson formulas once to that scalar. A non-finite participating
   error is a tile failure. On rejection, retain base `y`, update the common
   `h`, and rerun the tile trial. On acceptance, owner groups copy every
   participating candidate to base `y`, advance the shared `x`, and truncate
   the next `h` to the remaining common horizon.
4. **Epilogue** (owner warps): on successful completion, floor+normalize,
   `balance_charge`, `eos_re`, set `time = next_grid_time`, update
   `completed_steps = completed_global_steps + 1`, and update stats for
   participating cells. Cells stopped during setup store only their prescribed
   driver/stop state. A failed tile does not mark cells integrated or advance
   their time. `CellStateIO` applies those commit masks shared→global, then
   publishes `integrated_count` and `failure_code` through global atomics.

Notes:

- `dt_grid` is identical in every CTA for a global collapse step, but each CTA
  owns an independent ROS2S error controller and may use a different sequence
  of accepted substeps to reach that common horizon.
- Cells excluded by bounds, prior stopping, or the density-driver limit take
  neutral roles in reductions and never read/write out of bounds. They still
  execute every CTA barrier.
- The three RHS evaluations per substep are the *same code* — no
  instruction-cache divergence across k-stages.
- Base `y` is immutable for the entire tile trial. Candidate and stage buffers
  are separate, so CTA-wide rejection and singular retry are transactional.
- X/T and other trial scratch are sequenced by `barrier()`; v1 uses
  `barrier()` between all producer/consumer phases (FullBarrierSync).
- Any chemistry failure is terminal for the global run. Other CTAs may already
  have committed when the host observes the failure; v1 does not promise a
  grid-wide rollback or restartable failed state, and it never advances
  `grid_time` or writes a comparison result after such a failure.

### 4.3 DAG slicing (Singe "buffer" mode)

The source contains three generated scopes: `rhs_specie` (119 temporaries,
14 outputs), `rhs_eint` (162 temporaries, one returned output), and `jac_nuc`
(1,394 temporaries, 225 outputs). Temporary identifiers are of the form
`x<integer>_<integer>` (the suffix is not always `_0`), and many assignments
and outputs span multiple lines. The generator (§6) parses complete statements,
builds an explicit dependency graph, and verifies that source order is
topological; it does not infer convexity from line ranges.

For each 32-cell trial, every warp evaluates one generated slice across lanes
0..31. The generator assigns outputs to warps (J: balanced row groups; combined
RHS: balanced outputs), pulls in each output's transitive dependencies, and
applies Singe's recompute-vs-exchange rule:

- **Recompute** a shared subexpression in every warp that needs it when its
  subtree is cheap (pure arithmetic on already-shared values).
- **Exchange** expensive shared subexpressions — the transcendental families
  (`exp(a·log|T|)`, `exp(-a/T)`, `log`, `sqrt`, polynomial-in-log(T) powers):
  computed once per batch evaluation/region and staged through shared memory.

Slice code is emitted as ordinary Mojo functions with the same expression text
as the monolithic DAG. Each temporary is evaluated at most once per slice and
all exchanged values cross a barrier before consumption. The CPU oracle checks
bitwise identity for each J/RHS output, while GPU integration uses the final
tolerance policy.

Runtime `warp_id` dispatch means the aggregate instruction footprint includes
all generated paths. M2 therefore measures generated source/IR size and an
otherwise-identical 2/4/6/8-path dispatch microbenchmark before M4. The
generator must be able to split long slices into bounded regions with
CTA-uniform reconvergence points. If eight paths show instruction-fetch
regression, the fallback order is: rebalance/recompute to shorten regions,
split J into more reconvergent regions, then use a smaller feasible
`DAG_WARPS`. Eight long paths are not assumed safe without measurement.

### 4.4 Shared LU and solves (8-lane owner groups)

Per owned cell, 15×15 LU with partial pivoting (LINPACK-style, matching
`lu_decomposition`/`lu_solve` in `reproducer.cpp:96-181`):

- Each physical warp contains four independent 8-lane groups. In each group,
  local lane *l* owns shared columns/components `l` and `l+8` when present.
- At elimination step *k*, lane `k % 8` performs the ordered scan of
  `abs(E[i,k])`, `i >= k`, preserving first-strict-max tie-breaking. All lanes
  swap their owned shared columns, the pivot owner writes signed LINPACK
  multipliers, and owners of columns `c > k` apply the Schur updates in place.
- The 32 factorizations remain in the dynamic shared Jacobian arena through all
  three solves. RHS vectors are shared and distributed two components per lane;
  forward/back substitution reads the retained LU entries directly.
- CTA barriers synchronize the four groups in every warp and all eight warps.
  Participation masks guard memory operations only, so partial tiles follow the
  identical barrier sequence.

### 4.5 Data placement budget

V1 uses Hopper's large-shared-memory opt-in. The Jacobian/LU arena and the
generated-DAG scratch are dynamic shared memory because CUDA limits
non-opt-in static shared allocations to 48 KiB. The implemented layout is:

| Buffer | Formula | Bytes |
|---|---:|---:|
| Dynamic J/LU arena | `32 * 225 * 8` | 57,600 |
| Dynamic generated-DAG scratch | `32 * 534 * 8` | 136,704 |
| Dynamic RHS vectors | `32 * 15 * 8` | 3,840 |
| Dynamic X/T inputs | `32 * 15 * 8` | 3,840 |
| **Dynamic launch allocation** |  | **201,984** |
| Base `y` for the tile | `32 * 15 * 8` | 3,840 |
| Uncommitted candidate `y` | `32 * 15 * 8` | 3,840 |
| Trial vectors (stage-y, k1, k2, work) | `4 * 32 * 15 * 8` | 15,360 |
| LU pivots | `32 * 15 * 4` | 1,920 |
| Info, participation, error, controller | measured | 544 |
| **Static allocation (`ptxas`)** |  | **25,504** |
| **Total per CTA** |  | **227,488 (222.2 KiB)** |

`SmemLayout` is the source of truth: it computes aligned offsets and exposes a
comptime total. Kernel configuration validation rejects totals above 227 KiB.
The slicer emits 534 reusable `Float64` scratch slots per cell. The eight DAG
slices execute sequentially through that arena, with one physical warp
evaluating all 32 cells for each slice. This deliberately exchanges slice
parallelism for bounded live ranges and zero compiler spills. The table counts
base state only once and includes all trial vectors; there is no hidden
full-tile `BurnState` allocation. The measured layout leaves 4,960 bytes below
Hopper's 232,448-byte per-block shared-memory limit.

Private/register placement is measured rather than estimated as a hard
occupancy promise:

- LU factors, trial state vectors, all accept/reject candidates, and generated
  DAG temporaries live in shared memory. Boolean DAG temporaries are encoded
  as `0.0`/`1.0` in the numeric scratch arena.
- Physical register accounting, occupancy, spills, and private/local bytes come
  from the sm_90 compiler report. M3 records the LU-only report; M4
  records the full-kernel report. No source-level `InlineArray` assumption or
  guessed register cap substitutes for those reports.

The grid-timestep prepass additionally allocates global/host scratch for one
`Float64` minimum and one `Int32` cell ID per chemistry tile. This is outside
the CTA shared-memory budget. No per-cell timestep buffer is required: the
chemistry kernel recomputes `tff` from the prepared state.

### 4.6 Synchronization design (TilePipeline)

```mojo
# Interface sketch — NOT yet compiled
trait SyncStrategy(TrivialRegisterPassable):
    def phase_barrier(mut self): ...
    def producer_arrive(mut self, stage: Int): ...
    def consumer_wait(mut self, stage: Int): ...
    def consumer_release(mut self, stage: Int): ...
```

- **v1: `FullBarrierSync`** — `phase_barrier()` = `barrier()`; producer/
  consumer ops also map to `barrier()`. The phase schedule is static, and
  tests/audit must verify that masks and failure flags never bypass a barrier.
- **v2: `CounterSync`** — shared-memory atomic counters per stage with
  spin-wait (`Atomic.load` loop; first thread of each warp increments;
  on AMD emit `s_sleep 0` via `inlined_assembly` to yield), following the
  blog's AMD `SyncStrategy` design. Enables producer/consumer overlap
  (e.g. J slice production for batch b+1 overlapping LU of batch b).
- Acquire/release transitions are wrapped in context managers
  (`with stage.acquire() as buf:`) to prevent accidental unpaired stage
  transitions. This does not replace the CTA-uniform control-flow audit.

## 5. Structured components (Mojo interface sketches)

All sketches are **design intent, not compiled code**. Follow the
mojo-syntax skill: `def` only, `comptime`, `Self.`-qualified params,
`mut self`, traits, `raises` where needed.

```mojo
# ChemConfig — the comptime autotuning surface (Singe §4)
struct ChemConfig:
    comptime DAG_WARPS = 8
    comptime OWNER_WARPS = 8
    comptime TILE_CELLS = 32
    comptime BATCH_CELLS = 32
    comptime BATCHES_PER_TILE = 1
    comptime LU_GROUP_WIDTH = 8
    comptime LU_GROUPS_PER_WARP = 4
    comptime MAX_SHARED_BYTES = 100 * 1024
    comptime MAX_EXCHANGE_BYTES = 8 * 1024
    comptime Sync = FullBarrierSync           # v2: CounterSync
    comptime J_SLICE = jac_slice_tables       # from tools/slice_dag.py
    comptime RHS_SLICE = rhs_slice_tables

def validate_config[cfg: ChemConfig]():
    comptime assert cfg.TILE_CELLS == 32
    comptime assert WARP_SIZE == 32
    comptime assert cfg.BATCH_CELLS == cfg.TILE_CELLS
    comptime assert cfg.BATCHES_PER_TILE == 1
    comptime assert cfg.OWNER_WARPS * cfg.LU_GROUPS_PER_WARP == cfg.TILE_CELLS
    comptime assert cfg.LU_GROUP_WIDTH * cfg.LU_GROUPS_PER_WARP == WARP_SIZE
    comptime assert Neqs <= 2 * cfg.LU_GROUP_WIDTH
    comptime assert SmemLayout[cfg].BYTES <= cfg.MAX_SHARED_BYTES
```

```mojo
# CellStateIO (TileIO) — global <-> shared movement of cell state
struct CellStateIO[cfg: ChemConfig](TrivialRegisterPassable):
    def load(
        self,
        global_cells: DeviceCells,
        mut smem: SmemLayout,
        tile_idx: Int,
    ): ...
    def store(self, ...) -> None: ...
```

```mojo
# DagSliceOp (TileOp) — evaluate this warp's slice over all 32 cells
struct DagSliceOp[cfg: ChemConfig, table: SliceTable](TrivialRegisterPassable):
    def eval_tile(
        self,
        xt: SharedXT,            # batch X[14], T from shared
        mut out: SharedStage,    # J or ydot staging
        warp_id: Int, lane: Int,
    ): ...
```

```mojo
# SharedLuOp (TileOp) — four 8-lane systems per physical warp
struct SharedLuOp[cfg: ChemConfig](TrivialRegisterPassable):
    def factorize(self, mut j_lu: SharedJacobian, cell: Int, lane: Int) -> Int: ...
    def solve(self, j_lu: SharedJacobian, mut x: SharedVec, cell: Int, lane: Int): ...
```

```mojo
# Ros2sStepOp (TileOp) — tile-level step control; per-cell error norms in owner warps
struct Ros2sStepOp[cfg: ChemConfig](TrivialRegisterPassable):
    def error_norm(self, cell_slot: Int, ctrl: SharedControl) -> Float64: ...
    def tile_accept_or_reject(mut self, err_tile: Float64): ...
```

```mojo
# Grid-timestep prepass — perturb once and emit one minimum/cell pair per CTA
def prepare_grid_timestep_kernel[cfg: ChemConfig](
    cells: DeviceCells, num_cells: Int,
    completed_global_steps: Int, grid_time: Float64,
    step: Int, perturb: Bool,
    cta_dt_min: TileTensor[DType.float64, ...],
    cta_min_cell: TileTensor[DType.int32, ...],
    failure: UnsafePointer[Int32, MutAnyOrigin],
):
    ...

# Chemistry kernel — every CTA receives the same outer dt_grid
def chem_collapse_kernel[cfg: ChemConfig](
    cells: DeviceCells, num_cells: Int,
    completed_global_steps: Int, grid_time: Float64,
    next_grid_time: Float64, dt_grid: Float64,
    integrated_count: UnsafePointer[Int32, MutAnyOrigin],
    failure: UnsafePointer[Int32, MutAnyOrigin],
):
    validate_config[cfg]()
    var warp_id = thread_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE
    ref smem = ...  # shared layout from cfg
    var sync = cfg.Sync(smem)
    CellStateIO[cfg]().load(cells, smem, block_idx.x)
    sync.phase_barrier()
    driver_setup[cfg](
        smem, warp_id, lane, completed_global_steps, grid_time, dt_grid
    )
    sync.phase_barrier()
    while controller_running(smem):  # one CTA-uniform shared predicate
        begin_trial[cfg](smem, warp_id, lane)
        sync.phase_barrier()
        run_complete_tile_trial[cfg](smem, warp_id, lane)
        finish_trial[cfg](smem, warp_id, lane)  # singular retry or max-error decision
        sync.phase_barrier()
    epilogue[cfg](
        smem, warp_id, lane, completed_global_steps, next_grid_time
    )
    sync.phase_barrier()
    CellStateIO[cfg]().store(cells, smem, block_idx.x)
```

Every function that indexes a `TileTensor` must include the appropriate
`comptime assert tensor.flat_rank == N`; the ellipses above intentionally omit
concrete layouts and origins until M1 establishes them on the target GPU.

## 6. `tools/slice_dag.py` (generator tool)

**Input**: the machine-generated DAG functions in `reproducer.mojo`
(`rhs_specie`, `rhs_eint`, `jac_nuc`).

**Parse contract**:

1. Locate the three function bodies by name and indentation; names are scoped
   per function.
2. Collect complete Mojo statements with delimiter balancing, so multiline
   conditional expressions and multiline `mset`/`return` outputs remain
   intact. Recognize definitions matching `x[0-9]+_[0-9]+`; never assume the
   suffix is `_0` or that an assignment occupies one line.
3. Tokenize identifiers and build dependency edges to definitions in the same
   function. Whitelist external inputs/calls (`T`, `z`, `state.*`, `vget`,
   math helpers, constants); reject unresolved generated identifiers, duplicate
   definitions, cycles, use-before-definition, or unsupported statements. IR
   keys are `(function_name, temporary_name)` so the independent
   `rhs_specie`/`rhs_eint` namespaces cannot collide; generated slice helpers
   remain separately scoped (or use deterministic function prefixes).
4. Recognize all `vset(ydot, ...)`, `mset(jac, ...)`, and returned expressions
   as outputs, including multiline forms. Pin the current manifest at
   119/14 (`rhs_specie`), 162/1 (`rhs_eint`), and 1,394/225 (`jac_nuc`), so a
   regenerated network causes an intentional review rather than silent drift.
5. Preserve each expression's source text and stable source order. Embed the
   input-file hash and generator options in generated-file headers; two runs
   with identical inputs must have byte-identical output.

**Partition** (greedy, Singe §4.1 metrics — FLOP balance, register pressure,
locality):

1. Assign outputs to warps (J: row-blocks of the 15×15; RHS: round-robin).
2. Backward-reachability: each warp claims the transitive closure of its
   outputs' dependencies.
3. Temporaries claimed by ≥2 warps: mark **exchange** if subtree cost
   (FLOP-weighted; exp/log/cbrt weighted ~20–30) exceeds threshold τ,
   else **recompute** in each claiming warp. τ is a comptime-tunable knob.
4. Emit per-warp Mojo functions preserving expression text/order, comptime
   exchange tables (shared temporaries in topological order), and optional
   bounded dispatch regions with CTA-uniform reconvergence.

**Output**: `slices_jac.mojo`, `slices_rhs.mojo`, Python parser/generator tests
in `tests/test_slice_dag.py`, and a Mojo numerical validation driver in
`tests/test_slice_dag.mojo`. Also emit a machine-readable report containing
per-warp outputs, weighted operations, estimated peak live temporaries,
recomputed/exchanged nodes, exact exchange bytes, dispatch-region sizes, and
generated source bytes. M2 rejects any configuration above the 8 KiB exchange
budget before GPU compilation.

**CPU unit tests** (run anywhere): parser fixtures cover multiline assignments,
nonzero suffixes, malformed/unresolved references, and manifest drift. Sliced
evaluation must be **bitwise identical** to the monolithic functions over fixed
fixtures, deterministic log-scaled random states, and a reviewed branch-case
manifest. That manifest supplies `nextafter` neighbors where a condition maps
to a temperature threshold, and branch instrumentation proves that both sides
of every generated conditional execute somewhere in the corpus. These tests
must fail before the parser/slicer is written (TDD).

## 7. Testing plan (TDD)

Per Ben's rules: failing test first, minimal code to pass, refactor green.
Unit + integration + end-to-end tests. Tests that need a GPU are marked and
run only on the Linux+GPU machine; everything else runs on CPU anywhere.

| # | Test | Type | Where | Asserts |
|---|---|---|---|---|
| T0 | `tests/integration/test_grid_timestep_reference.mojo` | unit | CPU | stable `(dt, cell)` minimum, terminal `dt<10` update, density-limit stop, invalid-state failure, no-active termination, common density update, and exact common output time |
| T1 | `test_gpu_dataparallel_final_state.mojo` | e2e | GPU | one-thread-per-cell-trial Mojo baseline with the grid-timestep prepass and common CTA controller; grid-1 final state matches CPU Mojo within tolerance |
| T2 | `tests/test_slice_dag.py` + `.mojo` | unit | CPU | Python parser tests reject malformed/drifted input and prove deterministic generation; Mojo sliced J/RHS is bitwise-equal to monolithic on fixtures/log-random/X-floor cases, with reviewed `nextafter` cases and two-sided generated-branch coverage |
| T3 | `tests/gpu/test_warp_lu.mojo` | unit | GPU | column-owned LU+solve vs CPU LU on identity, exact-singular, forced-row-swap, pivot-tie, random, and near-singular 15×15 systems; residual/backward-error bounds hold |
| T4 | `test_structured_kernel_grid1.mojo` | integration | GPU | warp-specialized kernel grid-1 final state ≈ CPU reference |
| T5 | `test_structured_kernel_grid64.mojo` | e2e | GPU | grid-64 output passes the new grid-wide-policy reference comparator; wall time, code/resource data, and spills are recorded |
| T6 | `test_unified_controller.mojo` | integration | CPU+GPU | CPU/GPU CTA traces agree within tolerance; a rejection or singular LU in any batch discards every batch candidate, retries the whole CTA with one `h`, and all participants finish at `dt_grid` |
| T7 | `test_gridwide_timestep_multicta.mojo` | integration | GPU | limiter in a nonzero CTA and tied candidates choose the stable global pair; every CTA receives the same `dt_grid`; host time advances only after a successful launch with integrations |
| T8 | `test_partial_and_stopped_tiles.mojo` | integration | GPU | cell counts 1, 7, 8, 31, 32, and 33 plus mixed prior/during-setup stops have neutral masked lanes, no OOB access, identical barrier traces, and correct candidates/commits |

The packed comparison *schema and tolerances* from the C++ harness are reused,
but the legacy `final_states_grid64_cpu.bin` contents are not the reference for
the changed driver policy. M0 generates an explicitly named grid-wide CPU
reference (for example `final_states_grid64_cpu_gridwide.bin`) using scalar
RHS/J/LU code and the same fixed 32-cell CTA grouping. Legacy files are never
silently overwritten. Serialization must match the
`#pragma pack(1) PackedFinalState` layout: five little-endian `Int32` fields
(`cell, i, j, k, completed_steps`) followed by 19 little-endian `Float64`
fields (`time, density_driver, rho, T, e, xn[0..13]`), with no padding
(172 bytes/record). M0 includes a byte-for-byte C++↔Mojo round-trip fixture so
struct-layout assumptions cannot silently enter the file format. The new
reference also has a sidecar manifest recording cell count, `TILE_CELLS`,
driver/controller policy version, source hash, Mojo version, and tolerance
hash. `gridwide-cta32` comparison rejects a missing or mismatched manifest
rather than using a stale reference accidentally; `legacy-local` may retain
schema-only compatibility with the existing C++ file.

This harness work is real M0 scope: the current `reproducer.mojo` ignores
`--no-compare-final-state`, rejects `--compare-final-state` as unimplemented,
and does not emit packed multi-cell output. Implement the packed writer,
reader/comparator, PASS/FAIL+CSV reporting, and wall-time/multi-cell summaries
before T1/T5 claim reuse. T0/T6/T7 use the scalar grid-wide
driver/controller as the structural oracle. Extend the README results table
with backend=`mojo-gpu`, compiler resource metrics, prepass/reduction time,
the min/median/max distribution of finite **CTA minima**, and the limiting cell
ID.

M0 also makes the CPU CLI policy explicit without changing its existing
default: add `--driver-policy legacy-local|gridwide-cta32` (default
`legacy-local`) and `--write-final-state FILE`. Reference generation requires
`gridwide-cta32`; `--compare-final-state FILE` and
`--no-compare-final-state` control comparison independently. The GPU
executables implement only `gridwide-cta32` in v1 and reject any other policy.

**Performance protocol.** M1 and M4 time the same region: immediately before
the first preparation launch through the final chemistry synchronization,
including every host minima copy/reduction but excluding initialization,
reference-file I/O, and comparison. Run one untimed warm-up and report the
median of at least three grid-64 runs, plus preparation, host-reduction, and
chemistry subtotals. Record hardware, driver, Mojo version, build flags,
resource report, and work counters with each result; a wall-time comparison
without matching policy/input and these counters is not an acceptance result.

## 8. Milestones (in order)

**M0 — repo prep + grid-timestep reference.** Verify the pixi environment on
the GPU machine; implement the scalar CPU grid-wide timestep/32-cell-controller
reference, invariant checks, packed I/O/comparator, T0, and the CPU trace half
of T6. Generate the explicitly named grid-wide grid-64 reference only after
those tests pass, without replacing the legacy file. Check `std.gpu.sync` for
named-barrier/mbarrier availability and record findings in this doc, although
v1 does not depend on it.

**M1 — data-parallel Mojo GPU baseline.** Port `reproducer.mojo` to a
one-thread-per-cell *trial* kernel: each active thread retains the scalar
J/LU/state working set, but the 32-cell CTA uses the same common `h`, max-error
decision, transactional commit, and tile grouping as M4. Reuse scalar
RHS/J/LU bodies where GPU compilation permits, but split controller phases at
CTA barriers. Launch one physical warp per baseline CTA
(`block_dim = WARP_SIZE`, lanes 0..31 map to cells). Add the preparation
kernel, per-CTA `(minimum, cell)` buffers, host reduction, common density
update, and time invariants. Tests: T1, T7, T8, and the grid-64 comparator.
Record wall time, work counters, spills/private bytes, and prepass/reduction
overhead. **This is the numerically fair performance baseline and the harness
bring-up vehicle.**

**M2 — robust DAG slicer + dispatch feasibility.** Implement
`tools/slice_dag.py`, generated slices/reports, parser fixtures, and T2. Run the
2/4/6/8-path instruction-footprint microbenchmark on Hopper before
choosing the final dispatch-region layout. Reject generated configurations
whose exchange arena exceeds 8 KiB.

**M3 — column-owned LU + transactional trial microkernel.** Implement
`WarpLuOp`, T3, and the initial transactional trial path used by T6. M3's
portable four-batch schedule is superseded in M4 by the Hopper shared-LU
schedule after resource measurement showed that a 512-thread retained-column
kernel cannot launch with the generated DAG's register allocation.

**M4 — structured warp-specialized kernel.** Assemble components per §4–5
with `FullBarrierSync`, one 32-cell batch, dynamic shared J/LU, and four 8-lane
LU groups per warp. Tests: T4, T6, T7, T8, then T5. Capture the sm_90 resource
report and compare wall time/work counters against M1 before changing a default.

**M5 — tune.** Sweep only configurations that preserve `TILE_CELLS = 32` and
pass the layout assertions: feasible group widths, τ (recompute
vs exchange), dispatch-region size, and CTAs per compute unit. Then optionally
evaluate `CounterSync` overlap, constant striping across lanes (Singe §5.2),
warp indexing (Singe §5.3), and alternative shared-LU mappings. Re-run
correctness because controller grouping or reduction order
must not change silently; update the README table.

## 9. Numerics policy details

Allowed to differ from the CPU reference within harness tolerances:

- Reduction order in error norm, EOS sums, and density sums. LU uses the same
  ordered first-strict-max pivot rule, but downstream floating-point values may
  still differ across CPU/GPU math implementations.
- The outer driver uses
  `dt_grid = min_active(tff_reduc * tff)` and applies that same physical
  interval in every participating cell's density and chemistry update.
- Each CTA uses the maximum cell error for a common accepted/rejected ROS2S
  substep sequence. Different CTAs may take different internal sequences but
  must all finish at the same `dt_grid` horizon before the next global step.
- Out-of-bounds and stopped cells contribute neutral values to reductions and
  never commit state. Invalid participating physics is a failure, not a mask.
  Neither case alters the CTA barrier sequence.

Must NOT change:

- The per-temporary expression text/order inside DAG slices (slicer test
  enforces bitwise identity on CPU).
- ROS2S coefficients, controller parameters (`safe`, `fac_min`, `fac_max`,
  `uround`, `max_steps`), tolerance vectors (rtol/atol per component),
  LU-retry semantics (failure on the fifth singular factorization),
  perturbation schedule, the
  `density_driver += dt*density_driver/tff` rule (with `dt = dt_grid`), and
  the floor/normalize/balance-charge/eos sequence at step end.
- Stable outer-minimum tie-breaking by lowest global cell ID; fixed 32-cell CTA
  grouping in v1; transactional whole-tile retry/commit; exact assignment of
  integrated-cell time from host-computed `next_grid_time`.
- The final-state file format and comparison tolerances. The explicitly named
  grid-wide reference contents replace, rather than masquerade as, the legacy
  independent-timestep reference.

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Instruction-cache thrash from 8 structurally different J slices (Singe §5.1) | severe slowdown | M2 measures 2/4/6/8-path dispatch and generated code size; use bounded reconvergent regions, rebalance/recompute, or a smaller feasible warp count before M4 |
| Barrier overhead on small 15×15 shared LU | moderate | v1 accepts it; measure phase timings before introducing named-barrier overlap |
| Eight-lane groups own two columns/components per lane | moderate shared-memory traffic | all 32 cells remain concurrent; profile before considering a different group width |
| Generated DAG reaches 255 registers/thread | low occupancy | generated scalars use the explicit 534-slot shared arena; gate M4 on zero `ptxas` spill stores/loads and measured wall time |
| Shared layout exceeds Hopper opt-in after padding or generated scratch | launch/build failure | `SmemLayout.Bytes` comptime assertion, 227 KiB Hopper cap, exact `shared_mem_bytes`, and generator rejection before full compilation |
| Mojo stdlib missing named barriers | none for v1 | FullBarrierSync only needs `barrier()`; verify mbarrier API in M0 for v2 |
| Slicer mis-parses multiline/generated scopes | wrong results | balanced statement parser, pinned manifest/hash, negative fixtures, and bitwise T2 on every regeneration |
| Grid-wide minimum is dominated by one outlier cell | more outer collapse steps for the whole grid | this is the intended test-problem policy; report min/median/max over finite copied CTA minima and the limiting cell ID each global step |
| Preparation kernel + host pair reduction | extra launch, ~96 KiB transfer at grid-64, and synchronization per global step | prioritize an auditable v1 result; measure the isolated overhead in M1 and preserve stable `(dt, cell)` semantics in any later device reduction |
| Perturbation applied in both preparation and chemistry kernels | incorrect density/state update | preparation owns perturbation exclusively; T7 compares the prepared states and deterministic perturbation factors against the CPU reference |
| Unified-controller cost: a CTA advances at the pace of its worst cell's error estimate every substep (reject-if-any) | more substeps than per-cell adaptivity | record per-CTA accepted/rejected counts and compare M1 vs M4; tune tile composition only after correctness is established |
| A singularity occurs after candidate scratch was written | partial state update on retry | base `y` is immutable until the one CTA decision; T6 poisons candidate buffers and proves whole-tile discard/retry |
| One CTA fails after another CTA commits | globally partial failed state | failure is terminal in v1; host does not advance `grid_time` or compare/write final output, and no restart semantics are claimed |
| Legacy/stale final-state file is mistaken for the new-policy oracle | false failure or accidental baseline replacement | explicit `*_cpu_gridwide.bin` name, checked sidecar manifest, schema-only reuse, and no silent overwrite |

## 11. Reference commands

```bash
# environment (any machine)
pixi install
pixi run mojo --version

# CPU reproducer + baseline (runs anywhere)
pixi run build-mojo
pixi run run-mojo-grid1
c++ -std=c++20 -O3 -I. reproducer.cpp -o reproducer
./reproducer --grid 1 --no-compare-final-state

# planned tests (CPU parts, after M0/M2)
pixi run test-grid-timestep
pixi run test-slice-dag-python
pixi run test-slice-dag-mojo

# generate the new-policy CPU oracle (after M0)
./reproducer_mojo --grid 64 --driver-policy gridwide-cta32 \
  --write-final-state final_states_grid64_cpu_gridwide.bin \
  --no-compare-final-state

# GPU machine: baseline + structured kernel (after M1/M4)
pixi run build-gpu
pixi run build-structured-gpu
./reproducer_gpu --grid 1 --no-compare-final-state
./reproducer_structured_gpu --grid 64 \
  --compare-final-state final_states_grid64_cpu_gridwide.bin
```

### 11.1 Current Hopper evidence (2026-07-16)

- Environment: CUDA 12.9.1, driver 580.159.04, NVIDIA H200, compute
  capability 9.0, Mojo `1.0.0b3.dev2026071614` (`1084b3d8`).
- Grid 1: 438 global steps and final-state comparison PASS. The spill-free
  structured measured region is 8.707 s; this one-cell case masks 31 owner
  groups and is not a throughput result. Before explicit DAG scratch, the
  structured result was 4.917 s versus data-parallel 2.882 s.
- Grid 3 (27 cells, one partial CTA): structured 11.214 s versus data-parallel
  15.426 s, with 1,000 global steps in both runs.
- Before the CUDA driver was changed, the requested grid-32 comparison against
  legacy `reproducer.cpp` measured structured 29.926 s (1,000 grid-wide outer
  steps) versus CUDA 24.082 s (459 independent outer steps). That result was
  not equal work.
- After changing CUDA to use the same grid-wide outer timestep while retaining
  an independent ROS2S controller in each thread, grid 32 takes 14.781 s
  (14.93 s process time) for 1,000 outer steps. The structured kernel takes
  29.926 s (31.71 s process time) for the same 1,000 outer steps, a raw
  structured/CUDA ratio of 2.025. Controller behavior remains intentionally
  different: CUDA accepts/rejects and retries per cell, while structured uses
  one maximum-error CTA controller.
- Under the identical protocol at grid 64, modified CUDA takes 32.174 s
  (32.50 s process time) and structured takes 110.982 s (113.15 s process
  time), both for 1,000 common outer steps. The raw structured/CUDA integration
  time ratio is 3.449. This is one requested comparison run, not the
  median-of-three measurement required by the formal performance gate.
- Full adaptive sm_90 chemistry entry: 255 registers/thread, 25,504 bytes
  static shared, 201,984 bytes dynamic shared, a 360-byte intentional local
  frame, **0-byte spill stores, and 0-byte spill loads**. Every emitted
  Jacobian/RHS helper also reports a zero-byte stack frame and zero spills.
  The preparation entry uses 90 registers, 256 bytes static shared, a
  472-byte intentional local frame, and zero spills. These `ptxas` figures are
  static code-generation properties, not dynamic transaction counts.

## 12. Key file/line pointers

- `reproducer.mojo`: `rhs_specie` L257, `rhs_eint` L442, `jac_nuc` L1019,
  `lu_decomposition` L3829, `lu_solve` L3864, `integrate_ros2s` L3926,
  `burn_ros2s` L4064, `make_collapse_state` L4109, main/driver L4253+
- `reproducer.cpp`: LU L97/L155, ROS2S coefficients L7113, RODAS integrator
  L7133+, packed record L7481, collapse driver L7621, CUDA/HIP kernel L7671,
  final-state comparator L8225+, main/host loop L8398+
- `tools/transcendental_probe.py`: C++ vs Mojo Float64 transcendental
  comparison (use when chasing numerical differences)
- `README.md`: build/run commands + ROCm results table (extend with
  Mojo GPU rows in M5)

## 13. Decision log

| Date | Decision | By |
|---|---|---|
| 2026-07-16 | Implement in Mojo GPU directly (no CUDA/HIP prototype first) | Ben |
| 2026-07-16 | Tolerance-PASS acceptable; bitwise identity not required | Ben |
| 2026-07-16 | Milestone order M1→M5 (baseline first, then slicer, then structured kernel) | Ben |
| 2026-07-16 | Dev on macOS; GPU compile/test only on Linux+GPU machine (Metal lacks Float64) | environment constraint |
| 2026-07-16 | Grid-wide outer physical timestep `dt_grid = min_active(tff_reduc*tff)`; all integrated cells use it for density evolution, chemistry, and physical time | Ben |
| 2026-07-16 | Unified CTA-level ROS2S controller (single physical `h`, accept-if-all-accept); no per-cell horizons or controller state; only validity/stopped-cell masks remain | Ben |
| 2026-07-16 | Grid-wide synchronization ends at the outer `dt_grid` horizon; adaptive ROS2S substeps remain CTA-local | Ben |
| 2026-07-16 | M4 gives up gfx90a portability and targets Hopper large shared memory: one 32-cell batch, eight physical warps, four 8-lane LU groups per warp, and shared in-place factors | Ben |
| 2026-07-16 | Reuse the packed comparison schema/tolerances, but generate an explicitly named CPU reference for the new grid-wide/CTA-controller policy | design review |
| 2026-07-16 | Reduce stable `(dt_candidate, cell_id)` pairs and assign one host-computed `next_grid_time` to every integrated cell | design review |

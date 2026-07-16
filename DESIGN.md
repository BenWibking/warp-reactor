# Design: Warp-Specialized Mojo GPU Chemistry Kernel

Status: **approved design, not yet implemented**
Date: 2026-07-16
Authors: Ben Wibking + OpenCode session

This document is the complete, self-contained plan for rewriting the
primordial-chemistry reproducer (`reproducer.mojo` / `reproducer.cpp`) as a
**Singe-style warp-specialized GPU kernel in Mojo**, following Modular's
**structured kernel pattern** (TileIO / TilePipeline / TileOp). It is written
so a fresh session on the Linux+GPU machine can execute the plan without any
other context.

---

## 1. Goal and non-goals

**Goal.** Replace the current thread-per-cell GPU integration with a
warp-specialized kernel in which a CTA of *W* warps cooperates on a tile of
cells, eliminating the register spilling that dominates the current GPU
builds, while keeping the CPU and existing GPU entry points intact.

**Numerics policy (decided).** *Tolerance-PASS* against the existing
final-state comparison harness is acceptable. Bitwise identity with the CPU
reference is **not** required. Reductions (error norm, EOS sums, pivot
searches) may reassociate.

**Non-goals (v1).**

- No changes to the chemistry, EOS, or ROS2S numerics themselves.
- No CUDA/HIP rewrite (the C++ paths stay as-is; Mojo GPU is the new path).
- No persistent-megakernel driver; keep one kernel launch per global collapse
  step, matching the current CUDA driver structure.
- No producer/consumer overlap (v2); v1 uses full-CTA phase barriers.

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
- `TileTensor` is the common currency; warp roles are comptime-selected;
  context managers (`with producer.acquire() as tiles:`) make incorrect
  synchronization unrepresentable; everything is comptime-parameterized for
  autotuning.

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
- GPU targets: NVIDIA sm_90 and AMD gfx90a (per README). `WARP_SIZE` is
  comptime in Mojo: 32 on NVIDIA, 64 on AMD CDNA. The design uses
  `comptime WARP_SIZE` throughout; v1 accepts idle lanes on wave64
  (see §8 risks).
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

- **Tile**: one CTA of `NUM_WARPS = 8` warps (256 threads on NVIDIA)
  cooperates on `TILE_CELLS = WARP_SIZE` cells (32 on NVIDIA, 64 on AMD).
  Launch: `grid_dim = ceil(num_cells / TILE_CELLS)`,
  `block_dim = NUM_WARPS * WARP_SIZE`.
- **Lane ↔ cell**: for all bulk evaluation (DAG slices), lane *l* handles
  cell *l* of the tile.
- **Lockstep substepping**: every iteration of the substep loop executes
  exactly one ROS2S trial step (J → LU → k₁ → k₂ → k₃ → error) for every
  active lane, with per-lane `h` and per-lane accept/reject. Cells that
  finished (reached `tout` or failed) mask off; the tile loop runs until all
  cells in the tile finish. Each lane's trial-step sequence is deterministic
  given its own state, so per-cell trajectories match sequential execution
  (up to reduction reassociation, allowed by tolerance-PASS).
- The phase schedule per substep is **data-independent** ⇒ static schedule ⇒
  deadlock-free (Singe Theorem 1).
- **Cell ownership**: warp *w* additionally owns cells
  `{w, w+8, w+16, w+24}` (v1: 4 cells/warp) for all per-cell sequential work:
  LU factorization, triangular solves, state derivation, step control,
  collapse-driver bookkeeping.

### 4.2 Phase schedule (one kernel launch = one global collapse step)

0. **Load**: `CellStateIO` loads the tile's cell states (ρ, T, e, xn[14],
   time, density_driver, completed_steps, stats) global→shared.
1. **Driver setup** (owner warps, per owned cell): deterministic perturbation
   if scheduled for this global step; density-driver update and density
   rescale; build integrator state y[15] (number densities + e); set
   `tout = t + dt`. Mirrors `advance_collapse_step` in `reproducer.cpp`.
2. **Substep loop** (until all cells finish or a failure):
   - **P0 state-derive** (owner warps): floor xn, `eos_re` → write per-cell
     X[14] and T to shared for their owned cells. (Reused by J and k₁ RHS.)
   - **P1 Jacobian** (all warps, lane↔cell): each warp evaluates its slice of
     the 1,394-temporary J DAG for all TILE cells, writing its J entries to
     the shared J staging buffer in batches of `NUM_WARPS` cells.
     Owner warps then, per owned cell: form E = (1/(hγ))I − J from staging,
     warp-cooperative LU (§4.4) → factors in the warp's registers.
     LU-retry semantics (singular matrix ⇒ h /= 2, ≤5 retries) are
     owner-warp-local, exactly as in the sequential code.
   - **P2 k₁**: all warps evaluate RHS DAG slices (rhs_specie + rhs_eint,
     281 temporaries) → ydot[15] to shared; owner warps solve k₁ = E⁻¹ydot,
     form `ynew = y + a21·k1`, `ak2 = (c21/h)·k1`.
   - **P3 k₂**: owner warps derive X/T from ynew → shared; barrier; all warps
     RHS slices at (x + ct2·h, ynew); owners solve k₂, form
     `ynew = y + a31·k1 + a32·k2`, `work = (c31·k1 + c32·k2)/h`.
   - **P4 k₃ + step control**: owners derive X/T from ynew → shared; all
     warps RHS slices; owners solve k₃ (into work), form
     `ynew = y + b1·k1 + b2·k2 + b3·k3`, `err = e1·k1 + e2·k2 + e3·k3`;
     per-cell weighted error norm (warp reduction), accept/reject and new `h`
     with the existing Gustafsson controller — all owner-warp-local.
   - **P5 tile convergence**: shared flag/counter of unfinished cells;
     `barrier()`; loop.
3. **Epilogue** (owner warps): floor+normalize, `balance_charge`, `eos_re`,
   update time/density_driver/completed_steps/stats per owned cell;
   `CellStateIO` stores shared→global; tile-level reduction of
   `all_stopped`/`failure_code` into the existing global atomics.

Notes:

- Per-cell `dt` for the collapse step (from the free-fall driver) is
  per-lane; no cross-cell coupling anywhere except the tile loop count.
- The three RHS evaluations per substep are the *same code* — no
  instruction-cache divergence across k-stages.
- X/T shared buffers are double-buffered or sequenced by `barrier()`; v1
  uses `barrier()` between all phases (FullBarrierSync).

### 4.3 DAG slicing (Singe "buffer" mode)

The J DAG (1,394 temporaries, 225 outputs) and RHS DAG (281 temporaries,
15 outputs) are emitted by the code generator in topological order
(`var xN_0 = <expr>`), so contiguous ranges are valid convex partitions.
The `tools/slice_dag.py` generator (§6) assigns each output to a warp
(J: row-blocks ⇒ ~28–29 entries/warp; RHS: ~2 outputs/warp), pulls in the
temporaries each output transitively needs, and applies Singe's
recompute-vs-exchange rule:

- **Recompute** a shared subexpression in every warp that needs it when its
  subtree is cheap (pure arithmetic on already-shared values).
- **Exchange** expensive shared subexpressions — the transcendental families
  (`exp(a·log|T|)`, `exp(-a/T)`, `log`, `sqrt`, polynomial-in-log(T) powers):
  computed once per substep in a pre-pass and staged through shared memory.

Slice code is emitted as ordinary Mojo functions with the same expression
text as the monolithic DAG (per-temporary op order unchanged ⇒ slice
evaluation is bitwise-identical to monolithic on CPU, which the slicer's
unit test asserts).

### 4.4 Warp-cooperative LU and solves (owner warps)

Per owned cell, 15×15 LU with partial pivoting (LINPACK-style, matching
`lu_decomposition`/`lu_solve` in `reproducer.cpp:96-181`):

- Lane *c* ∈ [0,15) owns column *c* (15 doubles). Lanes 15..WARP_SIZE-1 idle
  in v1 (v2: process 2–4 cells concurrently per warp using 30/45/60 lanes).
- Pivot search: ordered scan of |E[i][k]|, i ≥ k, via `warp.shuffle` —
  tolerance-PASS permits any valid pivot choice; keep sequential
  first-strict-max tie-breaking anyway since it costs nothing.
- Row swap: shuffle exchange. Multipliers and Schur update: data-parallel
  across the 15 active lanes.
- LU factors (L unit-diagonal implied, U, ipvt) stay in the warp's registers:
  15 doubles/lane/cell ⇒ **60 registers/lane for 4 owned cells**. Never
  written to shared or global memory.
- Forward/back substitution (`lu_solve`): lane-cooperative with broadcasts;
  used 3× per substep (k₁, k₂, k₃), reusing the same factors.

### 4.5 Data placement budget

Shared memory per CTA (v1, TILE=32, Neqs=15):

| Buffer | Size |
|---|---|
| J staging (8-cell batch × 225 doubles) | 14.4 KB |
| ydot staging (32 × 15) | 3.84 KB |
| X/T state buffers (32 × ~18) | ~4.6 KB |
| Cell BurnState + collapse fields | ~6 KB |
| ROS2S control state per cell (h, reject, erracc, hacc, nsing, n_step, n_accept, x, tout) | ~4 KB |
| Sync counters/flags | <1 KB |
| **Total** | **≈ 33 KB** (gfx90a limit 64 KB; sm_90 limit 228 KB) |

Registers per lane (uniform allocation):

| Use | Budget |
|---|---|
| LU factors (4 cells × 15) | 60 |
| State vectors held transiently during solves (y/ynew/k1/k2/work) | ~20 |
| DAG slice transients (during P1/P2–P4) | ~40–80 live |
| **Target** | **≤128** (2 CTAs/SM); hard cap 168 |

`y`, `k1`, `k2`, `work`, ROS2S control state live in shared (cheap,
infrequently indexed); only LU factors and slice transients live in
registers.

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
  consumer ops also map to `barrier()`. Static phase schedule makes this
  correct by construction.
- **v2: `CounterSync`** — shared-memory atomic counters per stage with
  spin-wait (`Atomic.load` loop; first thread of each warp increments;
  on AMD emit `s_sleep 0` via `inlined_assembly` to yield), following the
  blog's AMD `SyncStrategy` design. Enables producer/consumer overlap
  (e.g. J slice production for batch b+1 overlapping LU of batch b).
- All acquire/release wrapped in context managers
  (`with stage.acquire() as buf:`) so unpaired sync is unrepresentable.

## 5. Structured components (Mojo interface sketches)

All sketches are **design intent, not compiled code**. Follow the
mojo-syntax skill: `def` only, `comptime`, `Self.`-qualified params,
`mut self`, traits, `raises` where needed.

```mojo
# ChemConfig — the comptime autotuning surface (Singe §4)
struct ChemConfig:
    comptime NUM_WARPS = 8
    comptime TILE_CELLS = WARP_SIZE          # 32 NVIDIA / 64 AMD
    comptime CELLS_PER_WARP = TILE_CELLS // NUM_WARPS  # 4
    comptime Sync = FullBarrierSync          # v2: CounterSync
    comptime J_SLICE = jac_slice_tables      # from tools/slice_dag.py
    comptime RHS_SLICE = rhs_slice_tables
```

```mojo
# CellStateIO (TileIO) — global <-> shared movement of cell state
struct CellStateIO[cfg: ChemConfig](TrivialRegisterPassable):
    def load(
        self,
        global_cells: TileTensor[DType.float64, ...],
        ref[AddressSpace.SHARED] smem: SmemLayout,
        tile_idx: Int,
    ): ...
    def store(self, ...) -> None: ...
```

```mojo
# DagSliceOp (TileOp) — evaluate this warp's slice, lane <-> cell
struct DagSliceOp[cfg: ChemConfig, table: SliceTable](TrivialRegisterPassable):
    def eval(
        self,
        xt: SharedXT,            # per-cell X[14], T from shared
        mut out: SharedStage,    # J or ydot staging
        warp_id: Int, lane: Int,
    ): ...
```

```mojo
# WarpLuOp (TileOp) — owner-warp LU + solves for owned cells
struct WarpLuOp[cfg: ChemConfig](TrivialRegisterPassable):
    var factors: InlineArray[Float64, 60]   # 4 cells x 15 columns
    var ipvt: InlineArray[Int, 60]
    def factorize(mut self, cell_slot: Int, fac: Float64, j_stage: SharedStage) -> Int: ...
    def solve(self, cell_slot: Int, mut x: SharedVec): ...
```

```mojo
# Ros2sStepOp (TileOp) — per-cell step control in owner warps
struct Ros2sStepOp[cfg: ChemConfig](TrivialRegisterPassable):
    def error_norm(self, cell_slot: Int, ctrl: SharedControl) -> Float64: ...
    def accept_or_reject(mut self, cell_slot: Int, ctrl: SharedControl, err: Float64): ...
```

```mojo
# Kernel skeleton — role dispatch is comptime-gated, phases are data-independent
def chem_collapse_kernel[cfg: ChemConfig](
    cells: TileTensor[DType.float64, ...], num_cells: Int,
    step: Int, perturb: Bool,
    all_stopped: UnsafePointer[Int32], failure: UnsafePointer[Int32],
):
    var warp_id = thread_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE
    ref smem = ...  # shared layout from cfg
    var sync = cfg.Sync(smem)
    CellStateIO[cfg]().load(cells, smem, block_idx.x)
    driver_setup[cfg](smem, warp_id, lane, step, perturb)
    while tile_has_unfinished_cells(smem):
        state_derive[cfg](smem, warp_id, lane)
        sync.phase_barrier()
        jac_phase[cfg](smem, warp_id, lane)     # slices + owner LU
        sync.phase_barrier()
        rhs_solve_phase[cfg](smem, warp_id, lane, stage=1)
        rhs_solve_phase[cfg](smem, warp_id, lane, stage=2)
        rhs_solve_phase[cfg](smem, warp_id, lane, stage=3)
        step_control_phase[cfg](smem, warp_id, lane)
        sync.phase_barrier()
    epilogue[cfg](smem, warp_id, lane)
    CellStateIO[cfg]().store(cells, smem, block_idx.x)
```

## 6. `tools/slice_dag.py` (generator tool)

**Input**: the machine-generated DAG functions in `reproducer.mojo`
(`rhs_specie`, `rhs_eint`, `jac_nuc`).

**Parse**: each `var xN_0 = <expr>` line; dependencies = `xM_0` tokens in
`<expr>` plus external inputs (`vget(X, i)`, `T`, `state.*`). Outputs =
`vset(ydot, i, ...)` / `mset(jac, i, j, ...)` / `return <expr>` statements.

**Partition** (greedy, Singe §4.1 metrics — FLOP balance, register pressure,
locality):

1. Assign outputs to warps (J: row-blocks of the 15×15; RHS: round-robin).
2. Backward-reachability: each warp claims the transitive closure of its
   outputs' dependencies.
3. Temporaries claimed by ≥2 warps: mark **exchange** if subtree cost
   (FLOP-weighted; exp/log/cbrt weighted ~20–30) exceeds threshold τ,
   else **recompute** in each claiming warp. τ is a comptime-tunable knob.
4. Emit per-warp Mojo functions preserving the original expression text and
   per-temporary order, plus comptime exchange tables (which temporaries go
   to shared, in topo order).

**Output**: `slices_jac.mojo`, `slices_rhs.mojo`, and a CPU validation
driver `test_slice_dag.mojo`.

**CPU unit test** (runs anywhere): for fixed representative inputs, sliced
evaluation must be **bitwise identical** to the monolithic functions
(same ops, same order per temporary). This test must exist and fail before
the slicer is written (TDD).

## 7. Testing plan (TDD)

Per Ben's rules: failing test first, minimal code to pass, refactor green.
Unit + integration + end-to-end tests. Tests that need a GPU are marked and
run only on the Linux+GPU machine; everything else runs on CPU anywhere.

| # | Test | Type | Where | Asserts |
|---|---|---|---|---|
| T1 | `test_gpu_dataparallel_final_state.mojo` | e2e | GPU | thread-per-cell Mojo GPU port, grid-1 final state matches CPU Mojo within tolerance |
| T2 | `test_slice_dag.mojo` | unit | CPU | sliced J/RHS eval bitwise-equal to monolithic on representative inputs (incl. T thresholds: T<2, ≤10, ≤30, ≤50, ≤100, ≤1000, ≤1160, ≤2000, ≤6000, ≤10000, >10000; X floors) |
| T3 | `test_warp_lu.mojo` | unit | GPU | warp-cooperative LU+solve vs CPU LU on random + near-singular 15×15 systems (tolerance) |
| T4 | `test_structured_kernel_grid1.mojo` | integration | GPU | warp-specialized kernel grid-1 final state ≈ CPU reference |
| T5 | `test_structured_kernel_grid64.mojo` | e2e | GPU | grid-64 final-state comparison harness PASS (tolerance) + wall-time/spill metrics recorded |
| T6 | `test_lockstep_equivalence.mojo` | integration | GPU | tile-lockstep substep trajectories ≈ per-cell sequential trajectories (tolerance) on a small grid with forced divergent h paths |

Existing harness reused: final-state binary comparison
(`final_states_grid64_*.bin`, PASS/FAIL + CSV), integrator stats counters,
wall-time output. Extend the README results table with Mojo GPU rows
(backend=`mojo-gpu`), including spill counts where the toolchain reports
them.

## 8. Milestones (in order)

**M0 — repo prep.** WIP branch `mojo-gpu-warp-spec`; commit pending
pixi.toml/pixi.lock (osx-arm64 platform); verify pixi env on the GPU
machine; check `std.gpu.sync` for named-barrier/mbarrier availability and
record findings in this doc.

**M1 — data-parallel Mojo GPU baseline.** Port `reproducer.mojo` to a
thread-per-cell GPU kernel reusing the existing scalar functions unchanged
(LayoutTensor stack allocations, InlineArray, std.math transcendentals are
expected to compile for GPU — verify). Driver mirrors the CUDA host loop.
Tests: T1. Record baseline wall time on the GPU machine. **This is the
performance baseline and the bring-up vehicle for the harness.**

**M2 — DAG slicer.** `tools/slice_dag.py` + generated slices. Test: T2 (CPU).

**M3 — warp-cooperative LU.** `WarpLuOp` + T3 (GPU).

**M4 — structured warp-specialized kernel.** Assemble components per §4–5
with `FullBarrierSync`. Tests: T4, T6, then T5. Perf compare vs M1.

**M5 — tune.** Comptime sweeps: NUM_WARPS ∈ {4, 8, 16}, τ (recompute vs
exchange), CTAs/SM, wave32 vs wave64 on AMD. Then optionally `CounterSync`
overlap, constant striping across lanes (Singe §5.2), warp indexing
(Singe §5.3), multi-cell-per-warp LU on wave64. Update README table.

## 9. Numerics policy details

Allowed to differ from the CPU reference within harness tolerances:

- Reduction order in error norm, EOS sums, density sums, pivot search.
- Idle-lane masking artifacts (must be none — masked lanes must not write
  shared or global state).

Must NOT change:

- The per-temporary expression text/order inside DAG slices (slicer test
  enforces bitwise identity on CPU).
- ROS2S coefficients, controller parameters (`safe`, `fac_min`, `fac_max`,
  `uround`, `max_steps`), tolerance vectors (rtol/atol per component),
  LU-retry semantics (≤5 halvings), perturbation schedule, density-driver
  update rule, floor/normalize/balance-charge/eos sequence at step end.
- The final-state file format and comparison tolerances.

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Instruction-cache thrash from 8 structurally-different J slices (Singe §5.1) | severe slowdown | slices are long straight-line code with few-way warp branches (acceptable per Singe); keep phase-level reconvergence; measure early in M4 with NUM_WARPS=2 first |
| Barrier overhead on small 15×15 LU (Singe diffusion lost ~15%) | moderate | v1 accepts it; v2 CounterSync + batching overlap |
| wave64 idle lanes on AMD (lanes 15–63 idle in LU) | moderate | v2: process 2–4 cells/warp concurrently; or compile wave32 kernels on gfx90a (comptime switch) |
| Register overshoot (>128) | occupancy 1 CTA/SM | move state vectors to shared (already planned), reduce CELLS_PER_WARP to 2 (NUM_WARPS=16), raise exchange threshold τ |
| Mojo stdlib missing named barriers | none for v1 | FullBarrierSync only needs `barrier()`; verify mbarrier API in M0 for v2 |
| Slicer mis-parses generated code | wrong results | T2 bitwise test is exact and cheap; run in CI on every regeneration |
| Lockstep divergence: cells with very different substep counts waste lanes | moderate | same waste exists per-warp today at warp granularity; optional future re-binning of cells into tiles by progress |

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

# tests (CPU parts)
pixi run mojo run test_slice_dag.mojo

# GPU machine: baseline + structured kernel (after M1/M4)
pixi run mojo build reproducer_gpu.mojo -o reproducer_gpu
./reproducer_gpu --grid 1 --no-compare-final-state
./reproducer_gpu --grid 64          # final-state comparison PASS expected
```

## 12. Key file/line pointers

- `reproducer.mojo`: `rhs_specie` L257, `rhs_eint` L442, `jac_nuc` L1019,
  `lu_decomposition`/`lu_solve` ~L3820–L3887, `integrate_ros2s` L3926,
  `burn_ros2s` L4064, `make_collapse_state` L4109, main/driver L4150+
- `reproducer.cpp`: integrator core L1–78, LU L79–186, network L187–7066,
  ROS2S coefficients L7118, RODAS integrator L7067–7330, collapse driver
  L7400+, CUDA kernel L7671, CUDA host loop L7694+
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

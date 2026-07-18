# Structured Hopper Optimization Results

Date: 2026-07-17

## Outcome

The fused base-state Jacobian and first-RHS DAG is retained. Its three-run
isolated H200 mean is **36.6046 s**, down from **38.2144 s** for the same Mojo
structured implementation, a **4.21% speedup**. This exceeds the revised 2.5%
keep threshold. The paired CUDA reference remains faster at **17.2347 s**;
that timing is a different executable with an independent per-cell inner
controller, not the timing of the Mojo structured kernel.

The scratch/occupancy redesign was rejected by resource feasibility analysis,
and an implemented independent per-cell controller was reverted after it
regressed the retained fused kernel by 1.13%. Jacobian caching was not
implemented during this pass; subsequent static analysis found that the
reported rejection counter omits startup rejections, so caching remains a
credible follow-up rather than a closed path. Approximate temperature rates
and explicit FMA experiments were not retained for the reasons below.

## Measurement Environment

All authoritative GPU measurements ran outside the sandbox in exclusive
one-H200 Slurm allocations. Each timing and profiling batch recorded the
assigned device's utilization, memory use, and compute processes before it
started. All assigned devices were at 0% utilization, 0 MiB used, with no
active compute process.

| Item | Value |
|---|---|
| GPU | NVIDIA H200 |
| Driver | 580.159.04 |
| Compute capability | 9.0 |
| CUDA module | CUDA/12.9.1 |
| CUDA toolkit | 12.9.86 |
| Mojo | 1.0.0b3.dev2026071705 (06cc3724) |
| Nsight Systems | 2025.1.3 |
| Nsight Compute | 2025.2.1 |
| Source commit | `c583ff0dbec237d26629fc955a64152b24dc8a14` plus the documented worktree changes |

The benchmark used grid 64, one warm-up, and three interleaved measured runs
of Mojo structured and CUDA. "Internal" is the application's GPU integration
timer; "process" is `/usr/bin/time -p` real time.

## Paired Performance

| Variant | Implementation | Internal runs (s) | Internal mean (s) | Process mean (s) | Decision |
|---|---|---:|---:|---:|---|
| Baseline | Mojo structured, CTA-wide controller | 38.2178, 38.2226, 38.2027 | 38.2144 | 40.4367 | Reference |
| Baseline pair | CUDA, independent controller | 17.4009, 17.4072, 17.4299 | 17.4127 | 17.9167 | Performance reference only |
| Fused base DAG | Mojo structured, CTA-wide controller | 36.6203, 36.6000, 36.5933 | **36.6046** | **38.6400** | **Retain: 4.21% internal, 4.44% process speedup** |
| Fused pair | CUDA, independent controller | 17.2378, 17.2284, 17.2380 | 17.2347 | 17.7167 | Performance reference only |
| Independent experiment | Mojo structured, per-cell controller | 37.0111, 37.0149, 37.0299 | 37.0186 | 39.0300 | Revert: 1.13% slower than fused |
| Independent pair | CUDA, independent controller | 17.4051, 17.4301, 17.3862 | 17.4072 | 17.8667 | Performance reference only |

The retained Mojo kernel is 2.124 times the paired CUDA time. Host overhead is
not the main gap: Nsight Systems attributes 36,267.1 ms of the retained run to
1,000 chemistry-kernel instances and only 63.7 ms to the preparation kernel.

Authoritative jobs and raw artifacts:

- Baseline: Slurm `12751722`, `benchmark_runs/baseline-12751722/`
- Fused: Slurm `12752255`, `benchmark_runs/fused-12752255/`
- Independent-controller experiment: Slurm `12753281`,
  `benchmark_runs/independent-12753281/`

## 1. Fused Jacobian and Base RHS: Retained

`tools/slice_dag.py` now constructs a combined DAG from `jac_nuc`,
`rhs_specie`, and `rhs_eint`, canonicalizes structurally identical
definitions across the three functions, and emits
`generated/slices_base_shared.mojo`. `src/gpu/structured_trial.mojo` prepares the base
state once and emits the Jacobian and first RHS together before LU
factorization.

Generator result:

| Quantity | Value |
|---|---:|
| Input definitions | 1,675 |
| Fused definitions | 1,113 |
| Eliminated definitions | 562 |
| Outputs | 240 |
| Base exchange values | 15 per cell |
| Fused scratch | 531 `Float64` values per cell |

Correctness and resource gates all passed:

- Grid 1 passed against the controller-compatible CPU oracle.
- All three grid-64 runs passed against the retained structured baseline with
  zero final-state difference reported by the comparator.
- PTXAS 12.9.1 reports 238 registers, a 360-byte intentional stack frame,
  **0-byte spill stores**, and **0-byte spill loads**.
- Static shared memory is 25,520 bytes and dynamic shared memory is 205,056
  bytes, within the H200 launch limit.

The targeted Nsight Compute sample also shows the intended code reduction:

| Metric | Baseline | Fused |
|---|---:|---:|
| Registers/thread | 255 | 238 |
| Dynamic shared/CTA | 200.45 KiB | 205.06 KiB |
| Executed instructions | 4,613,748 | 4,537,528 |
| One-or-more eligible cycles | 11.66% | 12.23% |
| Barrier stall share | 63.7% | 62.0% |
| Sample kernel duration | 3.30 ms | 3.09 ms |

## 2. Scratch and Resident-Warp Redesign: Rejected as Infeasible

The retained kernel is limited to one CTA per SM by **both** registers and
shared memory. Nsight Compute reports one-block register and shared-memory
limits, eight active warps/SM, and 12.5% achieved occupancy. Reducing only DAG
scratch therefore cannot admit a second CTA.

A generated 48-definition region schedule reduced fused scratch from 531 to
478 values per cell, saving 13,568 bytes per CTA. It would still use about
191.5 KiB dynamic plus 25.5 KiB static shared memory, far above the roughly
116 KiB total-per-CTA target for two CTAs, and the 238-register/thread limit
would independently keep occupancy at one CTA.

A generated 16-warp alternative increased fused scratch to 628 values per
cell (with 15 exchange values). Its shared-memory requirement exceeds the
launch headroom before accounting for the additional thread/register demand.
No GPU timing variant was built because neither generated schedule can change
the limiting occupancy state.

## 3. Independent Per-Cell Controllers: Implemented and Reverted

The experiment gave every eight-lane cell group independent `x`, `h`, error,
accept/reject history, singular retry state, and completion state while keeping
all CTA barriers uniform. It passed the grid-1 CPU comparison and completed a
64-cell local stress run.

The precondition was real: the baseline structured internal-step distribution
was 1,616 / 1,687 / 9,262 (min/median/max), while CUDA's was
1,365 / 1,543 / 8,799. Nevertheless, the structured CTA still executes until
its hardest cell completes, and SIMT lane masking does not remove the issued
DAG instruction stream. The authoritative mean was 37.0186 s, 1.13% slower
than the retained fused kernel. Registers also rose from 238 to 240 while
occupancy remained 12.5%, so the implementation was reverted.

As expected for a different adaptive controller and arithmetic ordering, the
experiment was not bitwise identical to CUDA. A 64-cell comparison observed a
maximum 0.393% thermodynamic relative difference and 2.59% non-deuterium
species relative difference; the existing cross-implementation comparator
tolerances were exceeded. This was recorded but was not the reason for the
performance rejection.

## 4. Rejected-Step Jacobian Cache: Not Implemented; Reconsider

The retained fused grid-64 *counted* rejection distribution is 0 / 0 / 295
against an internal-step distribution of 1,616 / 1,687 / 9,262. That counter
only increments after the first accepted inner step, so it omits startup
rejections and is not a valid cache-frequency proxy by itself.

The saved CUDA counters expose the omission: CUDA reports internal/Jacobian
counts of 1,365/1,217, 1,543/1,321, and 8,799/6,553. Its source evaluates the
raw Jacobian outside the retry loop and reconstructs/refactors the shifted
matrix after a rejection. Mojo reports identical internal/Jacobian counts
because it regenerates the fused base DAG on every attempt. Thus CUDA reuses
the Jacobian on about 14% of median attempts and 26% of maximum attempts.

No cache implementation or timing was performed in this optimization pass.
The earlier skip decision was based on incomplete instrumentation and should
not be treated as evidence that a cache cannot amortize. A future design would
need to balance the saved 225-output Jacobian DAG against the storage/reload
cost without assuming that the counted rejection statistic is complete.

## 5. Temperature-Rate Work

The exact, low-risk levels were implemented as part of the retained fused DAG:
syntactic CSE now crosses Jacobian/RHS boundaries, and the fused shared
producer exchanges 15 profitable base-state values per cell. This is where
the 562 eliminated definitions come from.

Piecewise polynomial or table approximations were not introduced. They would
change the chemical rate model, require a separate direct-rate error campaign,
and are not justified while the exact CSE implementation already clears the
keep threshold. This remains future work rather than an unmeasured retained
change.

## 6. Controlled FMA: Not Implemented

Nsight Compute reports 509,888 fused and 500,698 non-fused FP64 instructions
for the retained kernel, but estimates only a 1.51% local speedup even if the
available non-fused pairs were converted. That is below the 2.5% keep
threshold before accounting for changed rounding and comparator risk.
Consequently no explicit-FMA source variant was retained.

## Validation Summary

- `python3 -m unittest tests.test_slice_dag`: 11 passed.
- `pixi run mojo run -I . tests/test_slice_dag.mojo`: 1 passed.
- `pixi run mojo run -I . tests/test_summary.mojo`: 2 passed.
- `pixi run test-grid-timestep`: 9 passed.
- Retained structured GPU executable built successfully for H200.
- Grid-1 GPU/CPU final-state comparison passed.
- Grid-64 fused/baseline comparisons passed on all three measured runs.
- PTXAS: zero spill stores and zero spill loads.
- `git diff --check`: passed.

The reusable paired benchmark is `benchmarks/paired_h200_baseline.slurm`; it
records the environment and GPU state, runs the warmed/interleaved timing
sequence, and collects Nsight Systems and targeted Nsight Compute reports.

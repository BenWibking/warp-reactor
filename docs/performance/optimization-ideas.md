# Structured Hopper Kernel Optimization Ideas

This document records the remaining credible paths for making the structured
Mojo chemistry kernel outperform the grid-wide independent CUDA reference on
Hopper. The outer grid controller remains host-resident. GPU-side reduction is
allowed, but moving the complete outer controller onto the device is out of
scope.

## Current Baseline

The retained grid-64 structured kernel completed two isolated H200 runs in
38.2381 s and 38.2609 s, for a mean of 38.2495 s. The most recent CUDA
reference measurement was 32.174 s, although it was not collected in the same
isolated Slurm job. On those numbers, the structured implementation needs at
least a 16% reduction in integration time to win. A paired isolated benchmark
should establish the exact current gap before applying another optimization.

Current resource usage for the full adaptive chemistry entry is:

- 255 registers per thread.
- 25,520 bytes of static shared memory.
- 200,448 bytes of dynamic shared memory.
- A 360-byte intentional local stack frame.
- Zero spill stores and zero spill loads.

The dynamic allocation consists of 32 cells multiplied by 783 `Float64`
values per cell:

| Region | Values per cell | Bytes per CTA |
|---|---:|---:|
| Jacobian | 225 | 57,600 |
| DAG scratch | 515 | 131,840 |
| RHS | 15 | 3,840 |
| DAG inputs | 15 | 3,840 |
| RHS exchange | 13 | 3,328 |
| **Total** | **783** | **200,448** |

Together, the static and dynamic allocations consume nearly the full Hopper
shared-memory allowance, so only one eight-warp chemistry CTA can reside on an
SM.

## Retained Improvements

- Process 32 cells per CTA using Hopper-sized shared memory.
- Use four eight-lane LU groups per physical warp.
- Replace CTA barriers inside independent LU groups with warp synchronization.
- Keep the generated kernel at zero spills.
- Balance the generated DAG as a three-warp first wave and five-warp second
  wave.
- Compute and exchange 13 profitable RHS values once per cell.
- Encode active cell identity with the participant state to avoid redundant
  indexing work.

## Rejected or Exhausted Experiments

- Counter-based producer/consumer synchronization passed correctness and kept
  zero spills, but regressed isolated grid-64 time from 38.2495 s to 40.1515 s,
  or 4.97%.
- Alternative warp indexing and constant-placement experiments did not improve
  the retained kernel.
- Additional region splitting and scratch-packing variants either increased
  resource pressure, failed the launch limit, or did not improve performance.
- Eliminating register spills is complete; PTXAS reports zero spills.
- Moving the outer grid controller onto the device is intentionally excluded.

## Ranked Remaining Optimizations

### 1. Fuse the Base-State Jacobian and RHS

Each Rosenbrock attempt evaluates the Jacobian and then the first RHS using the
same base state. Generate one combined DAG that produces both results.

The combined generator should perform common-subexpression elimination across
`jac_nuc`, `rhs_specie`, and `rhs_eint`, especially for:

- `log(T)`, `sqrt(T)`, reciprocal temperature, and temperature powers.
- Temperature-dependent reaction-rate coefficients.
- Density, species, and thermodynamic intermediates shared by the Jacobian and
  RHS.

This removes duplicated transcendental work, one DAG input-preparation pass,
and potentially one CTA phase boundary. It preserves the current controller
and precision model, so it is the preferred next implementation.

Success criteria:

- Final-state comparison passes at grid 1 and grid 64.
- PTXAS still reports zero spills.
- Shared memory remains within the Hopper launch limit.
- Isolated grid-64 mean improves by at least 2.5%; otherwise revert.

### 2. Reduce DAG Scratch and Increase Resident Warps

DAG scratch consumes 131,840 bytes, or about 58% of total per-CTA shared
memory. Reducing it is the only realistic route to materially higher
occupancy.

Possible placement policy:

- Keep only values live across generated regions in shared memory.
- Keep short-lived region temporaries in registers.
- Recompute cheap arithmetic instead of persisting it.
- Place cold, long-lived values in a coalesced global workspace when the extra
  memory traffic costs less than the gained latency hiding.

Two 32-cell CTAs per SM require total shared memory per CTA to fall to roughly
half the current allocation. That is a difficult target because the Jacobian
and non-scratch state already consume about 94 KiB including static shared
memory. A hybrid register/shared/global schedule should therefore be evaluated
instead of attempting another shared-only packing pass.

An alternative is a 16-warp DAG partition within one CTA. It increases
resident warps without duplicating the cell batch, but must avoid excessive
closure duplication and shared scratch growth.

### 3. Independent Per-Cell Inner Controllers

Retain the host-resident grid-wide physical timestep, but give each cell its
own inner ROS2S state:

- Current integration position `x`.
- Trial step `h`.
- Error and rejection state.
- Completion and active masks.

The generated DAG continues to operate over active lanes. Cells that finish
early become inactive while harder cells continue. This avoids making all 32
cells follow the worst cell's error and rejected-step history and more closely
matches the CUDA reference algorithm.

Before implementation, record per-cell internal steps, rejected steps, RHS
calls, and Jacobian calls for representative grid-64 CTAs. If those counts have
little dispersion, this optimization will only add divergence and state.
Cells can also be bucketed by a stiffness proxy before CTA assignment to reduce
masked-lane waste.

### 4. Cache the Jacobian Across Rejected Attempts

A rejected Rosenbrock attempt changes `h`, but the base state and its chemical
Jacobian remain unchanged. The current in-place factorization destroys the raw
Jacobian, forcing it to be regenerated.

Store the unfactored 15-by-15 Jacobian in a global workspace before
factorization. On rejection, reload it, update the `1 / (h * gamma)` diagonal,
and refactor it. H200 memory capacity and bandwidth make a roughly 472 MiB
grid-64 cache practical, but the optimization is useful only when rejected
steps occur often enough to amortize the extra store.

Instrument rejection frequency first. Skip this work if rejected attempts are
rare.

### 5. Optimize Temperature-Dependent Rates

The generated network repeatedly evaluates expressions such as
`exp(a * log(T))`. CUDA performs the same analytic work, so reducing it provides
a way to create an algorithmic advantage.

Increasing levels of aggressiveness are:

1. Syntactic common-subexpression elimination across slices.
2. A shared producer for frequently reused temperature functions and rate
   coefficients.
3. Piecewise polynomial approximations over the physically visited
   temperature range.
4. Temperature-indexed rate and derivative tables with accurate interpolation.

Every approximation must be checked against direct rate evaluations and the
grid-64 final-state comparator. Table resolution and polynomial degree should
be selected from the required chemistry tolerance rather than machine epsilon.

### 6. Selectively Restore FMA Contraction

`portable_mul` deliberately emits separately rounded FP64 multiplication on
NVIDIA. This inhibits contraction in accumulation expressions. Replace only
well-defined multiply-add chains with explicit FMA and compare numerical drift.

Start with ROS2S vector updates and Jacobian/LU accumulations, where contraction
removes a separate add instruction. Avoid a global replacement until each
category has independent correctness and performance results. This is expected
to be a single-digit improvement and is not sufficient by itself.

## Lower-Priority Options

- Hardware named barriers or `mbarrier` may reduce synchronization overhead,
  but the CounterSync result indicates that phase synchronization is not large
  enough to justify another synchronization-first experiment without profiler
  evidence.
- Mixed-precision LU with iterative refinement could reduce factorization cost
  and shared storage, but adds substantial correctness risk and does not reduce
  the dominant FP64 DAG scratch unless intermediates are also compressed.
- Tensor-core LU is possible for the 15-by-15 systems, but the small pivoted
  matrices and transcendental-heavy chemistry make it a lower-priority rewrite.
- Device-side scalar reductions can reduce host mapping work while keeping the
  outer controller on the host. The CUDA reference already performs larger
  host transfers, so this is unlikely to close the kernel performance gap by
  itself.

## Measurement Plan

Before the next implementation, submit a paired one-H200 Slurm job containing
the retained structured binary and the CUDA reference. Record:

- Three warmed, interleaved grid-64 runs for each implementation.
- CUDA module, driver, GPU model, compute capability, and initial utilization.
- Internal integration time and process wall time.
- Internal step, rejection, RHS, Jacobian, decomposition, and solve counts.
- An Nsight Systems kernel/API breakdown.
- A targeted Nsight Compute report for the chemistry kernels, including
  eligible warps, issue stalls, barrier stalls, shared-memory stalls, achieved
  occupancy, and FP64/transcendental instruction counts.

Apply optimizations one at a time. Keep a change only when it passes numerical
comparison, preserves zero spills unless a measured tradeoff justifies them,
and improves the isolated paired benchmark.

## Recommended Sequence

1. Establish a paired isolated CUDA/structured baseline and collect controller
   statistics.
2. Implement the fused Jacobian plus base-RHS DAG with cross-function CSE.
3. Use the profile and generated liveness data to redesign scratch placement.
4. Implement independent per-cell inner controllers if step-count dispersion
   supports them.
5. Cache rejected-step Jacobians if rejection frequency supports it.
6. Evaluate controlled FMA and temperature-rate approximations.

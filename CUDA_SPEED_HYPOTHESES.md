# Why the Original CUDA Kernel Is Still About Twice as Fast

Date: 2026-07-18

## Scope and Confidence

This analysis now incorporates the requested Nsight Compute comparison of the
production grid-64 workload. Nsight Compute 2025.2.1 profiled full-size launch
indices 0, 499, and 999 for each implementation with the full metric set and
strict application replay. Each selected structured launch had 8,192 CTAs of
256 threads; each CUDA launch had 2,048 CTAs of 128 threads. The raw reports,
plain-text exports, environment record, and idle-GPU snapshots are in
`benchmark_runs/ncu-grid64-12754915/`.

Slurm job 12754915 completed with exit code 0. It ran on an NVIDIA H200
(compute capability 9.0) with driver 580.159.04, CUDA 12.9.1, and Nsight
Compute 2025.2.1. Before every profile the allocated GPU reported 0%
utilization, 0 MiB used, and no active compute process. The only profiler
warning was that six C2C-link traffic metrics were unavailable; none is used
here.

The short version is that CUDA is faster **despite** its spills, not because
the spills are harmless or beneficial. In the middle and late full-grid
launches, structured Mojo executes 3.05 times as many warp instructions and
approximately 2.47 times as many predicated-on thread instructions as CUDA.
CUDA finishes those launches 1.89 and 1.84 times faster. Both kernels have the
same 12.5% theoretical occupancy and both leave the scheduler without an
eligible warp about 90% of cycles. Scheduler starvation is therefore a common
condition, not the differentiator that the earlier Mojo-only sample suggested.

Mojo spends about 62% of cycles per issued instruction at CTA barriers; CUDA
has no barrier stalls, but instead spends about 49% on instruction fetch and
26% on long-scoreboard dependencies in the representative middle launch.
CUDA's spilling generates enormous local-memory traffic, but reaches only 14%
of peak memory throughput. Its much lower instruction count, nearly full-warp
execution, Jacobian reuse across retries, and absence of shared-memory
materialization outweigh that spill penalty.

Two meanings of "independent" must be kept separate:

- **Independent numerical substepping:** each cell owns `x`, `h`, error, and
  rejection history. The controlled Mojo experiment indicates that this is
  not a speedup by itself and can be a regression.
- **Independent execution ownership:** a CUDA thread owns its cell's complete
  RHS/Jacobian/LU path without cross-warp producer/consumer barriers. This
  remains a leading architectural hypothesis.

## Relevant Observations

| Property | Original CUDA | Retained structured Mojo |
|---|---:|---:|
| Grid-64 mean | 17.2347 s | 36.6046 s |
| Chemistry-kernel time in saved Nsight Systems run | 15,767.2 ms | 36,267.1 ms |
| Work mapping | One complete cell per CUDA thread | 32 cells cooperatively processed by eight warps |
| Production launch | 2,048 CTAs x 128 threads | 8,192 CTAs x 256 threads |
| Registers/thread | 255 | 238 |
| Runtime stack size reported by NCU | 6,304 bytes/thread | 1,024 bytes/thread |
| PTXAS spill report | 6,290 B stores, 9,408 B loads | 0 B stores, 0 B loads |
| Explicit CTA barriers | 0 according to PTXAS | Used throughout every ROS2S attempt |
| Chemistry shared memory | 0 bytes/block | 205.06 KiB dynamic + 25.52 KiB static/block |
| Resident blocks and warps/SM | 2 blocks, 8 warps | 1 block, 8 warps |
| Middle achieved occupancy | 12.10% | 12.50% |
| Middle no-eligible scheduler cycles | 89.96% | 89.38% |
| Middle issue slots busy | 9.95% | 10.62% |
| Middle dominant stall | No instruction, 49.43% | CTA barrier, 62.34% |
| Middle executed warp instructions | 0.632 billion | 1.925 billion |
| Middle branch efficiency | 99.65% | 79.75% |
| Middle DRAM bytes read + written | 7.80 GB | 97.03 MB |
| Middle memory throughput / peak | 492.70 GB/s / 14.02% | 3.24 GB/s / 3.53% |
| Middle profiled-launch duration | 15.828 ms | 29.936 ms |

The 2.12-times application ratio actually understates the chemistry-kernel
ratio: the saved Nsight Systems totals differ by 2.30 times. CUDA has more
non-kernel overhead in that run, so launch, reduction, and host bookkeeping
cannot explain Mojo's deficit.

The earlier grid-4 report remains useful as a consistency check, but its
87.77% no-eligible value should be superseded by the production-grid values
below. The full-grid reports also remove the previous need to infer CUDA's
occupancy, spill hit rates, and scheduler stalls. The hypotheses still overlap
and must not be added as an independent percentage decomposition.

### Phase consistency

| Launch | Mojo duration | CUDA duration | Duration ratio | Mojo/CUDA warp-instruction ratio | Mojo/CUDA predicated-on thread-instruction ratio |
|---|---:|---:|---:|---:|---:|
| 0, early | 214.990 ms | 72.074 ms | 2.983x | 3.426x | 2.784x |
| 499, middle | 29.936 ms | 15.828 ms | 1.891x | 3.048x | 2.474x |
| 999, late | 23.147 ms | 12.598 ms | 1.837x | 3.048x | 2.473x |

These durations are for one selected chemistry-kernel launch, not application
wall time or Nsight Compute's replay wall time. Application replay ran the
complete grid-64 program on every metric pass, while `--launch-count 1`
recorded only the selected production-size launch.

The thread-instruction values are derived, not direct counters: executed warp
instructions are multiplied by NCU's average number of predicated-on threads
per executed warp instruction. The ratio is stable in the middle and late
launches and tracks the observed duration ratio much better than occupancy or
scheduler eligibility does.

## Primary Finding: CUDA Executes Much Less Issued Work

Confidence: **high; directly measured**.

For launch 499, Mojo executes 1,925,378,110 warp instructions while CUDA
executes 631,644,751, a factor of 3.05. CUDA's instructions also make better
use of each issued warp: it averages 31.89 active and 30.40 predicated-on
threads per instruction, compared with Mojo's 26.38 and 24.67. Multiplying
through gives approximately 50.8 billion active-thread instructions for Mojo
versus 20.1 billion for CUDA, a factor of 2.52. The predicated-on ratio is
2.47.

This is not merely a different accounting convention caused by Mojo launching
eight times as many threads. NCU counts instructions actually executed by the
SMs. The cooperative implementation succeeds in sharing some work across its
eight threads per cell, but its generated DAG handoffs, role checks,
cooperative LU, reductions, synchronization, and controller still leave it
with much more executed work. The FP64 subset alone is 424.1 million
instructions for Mojo versus 264.1 million for CUDA, a 1.61-times difference.
Mojo additionally executes 175.7 million shared load/store instructions in
the middle launch; CUDA executes none.

Control efficiency points in the same direction. Middle-launch branch
efficiency is 79.75% for Mojo and 99.65% for CUDA. CUDA's nearly uniform
branches and 30.40 predicated-on lanes contradict the idea that independent
cell histories are causing a large divergence penalty in the original kernel
at this phase. They do not show that independent substepping is intrinsically
faster; the controlled Mojo intervention still regressed.

The 24.67-versus-30.40 predicated-on-lane comparison should not be read as a
large difference in compiler predication alone. Mojo falls from 32 possible
lanes to 26.38 active lanes, then from 26.38 to 24.67 when false instruction
predicates are removed. CUDA falls from 32 to 31.89, then to 30.40. Thus 5.51
of the 5.73-lane gap, about 96%, comes from the active mask; only 0.22 lane per
warp instruction comes from the difference in false predicates.

The full-warp DAG role tests are warp-uniform and do not create this active-mask
loss on a full 32-cell tile. The structured kernel is not purely warp-granular
outside that phase, however. Its LU maps four cells to four eight-lane groups
inside each physical warp. Pivot selection and scaling select one lane per
eight-lane group, triangular loops use shrinking lane subsets, per-cell error
norms select each group's lane zero, and CTA controller loops sometimes select
only thread zero. Existing SASS source counters show the hottest pivot-search
region executing with four active lanes and repeated solve regions averaging
30 active but only 14 predicate-true lanes. This intra-warp cooperative work,
not physical-warp DAG specialization itself, explains most of the lane
utilization difference.

## Finding: Barrier Stalls Are Real, but Eligibility Is Not the Difference

Confidence: **high; directly measured**.

The production-grid profile confirms that barriers dominate Mojo's per-issued
instruction latency. In launch 499, each warp spends 11.733 of 18.821 cycles
per issued instruction stalled at a CTA barrier, or 62.34%. Launches 0 and 999
show 60.83% and 62.43%. This is not a tiny-grid artifact.

However, CUDA is not better at keeping the scheduler eligible:

| Scheduler metric, launch 499 | CUDA | Structured Mojo |
|---|---:|---:|
| Active warps/scheduler | 1.97 | 2.00 |
| Eligible warps/scheduler | 0.106 | 0.113 |
| Scheduler cycles with no eligible warp | 89.96% | 89.38% |
| Warp cycles/issued instruction | 19.62 | 18.82 |
| Issue slots busy | 9.95% | 10.62% |

CUDA is marginally worse on every one of these aggregate scheduler measures.
Its missing barrier stalls are replaced mainly by 9.701 cycles with no
instruction ready to fetch (49.43%), 5.155 long-scoreboard cycles (26.27%),
and 2.685 fixed-latency wait cycles (13.69%). The likely interpretation of the
first two is pressure from CUDA's very large instruction body and local/global
memory dependencies, respectively.

The correct causal statement is therefore narrower than the original
hypothesis: barriers are a major component of Mojo's cost per instruction and
constrain compiler scheduling across phases, but CUDA wins primarily because
it needs far fewer instructions, not because it exposes more eligible warps.
The structured DAG's three-warp/five-warp waves, CTA-wide phase boundaries,
and synchronized cooperative LU remain plausible sources of that extra work
and waiting. A 62% stall fraction is not a prediction that removing barriers
would make the whole application 62% faster.

## Primary Hypothesis: CUDA Performs Fewer Expensive Jacobian Evaluations

Confidence: **high**.

The saved counters show a larger algorithmic difference than the counted
rejection statistic initially suggested:

| Counter, min/median/max | CUDA | Structured Mojo |
|---|---:|---:|
| Internal attempts | 1,365 / 1,543 / 8,799 | 1,616 / 1,687 / 9,262 |
| Jacobian calls | 1,217 / 1,321 / 6,553 | 1,616 / 1,687 / 9,262 |
| RHS calls | 4,095 / 4,629 / 26,397 | 4,848 / 5,061 / 27,786 |
| Accepted steps | 1,217 / 1,321 / 6,553 | 1,376 / 1,431 / 6,785 |

CUDA evaluates the chemical Jacobian once before entering its retry loop. If
an error rejection changes only `h`, it rebuilds and refactors the shifted
matrix from the preserved raw Jacobian without reevaluating the chemistry.
Consequently its Jacobian-call count equals its accepted-step count, not its
attempt count. CUDA reuses the Jacobian on about 14% of median attempts and
26% of maximum attempts.

Mojo evaluates the fused base DAG, including all 225 Jacobian outputs, at the
start of every attempt. Its Jacobian-call count therefore equals its internal
attempt count. Relative to Mojo, CUDA performs about 22% fewer median Jacobian
evaluations and 29% fewer maximum Jacobian evaluations. The Jacobian is much
larger than the 15-component RHS, so this can be a substantial fraction of
the unexplained gap.

This also corrects a misleading statistic. Both implementations increment the
reported rejection counter only after an inner integration has already
accepted at least one step. Startup rejections are omitted. For Mojo, the
difference between internal and accepted steps is 15% at the median and 27%
at the maximum even though the reported median rejected-step count is zero.
The earlier zero median was therefore not evidence that Jacobian reuse was
irrelevant.

## Counter Observation: CUDA's Independent Controller Does Less Logical Work

Confidence: **high for the counter difference, low that numerical substep
independence is itself a positive performance mechanism**.

CUDA's independent controller uses 8.5% fewer median attempts and 5.0% fewer
maximum attempts than the CTA-wide Mojo controller. Mojo takes the maximum
error across every participating cell in a 32-cell tile, so easy cells inherit
the hardest cell's step size and rejection history.

However, the controlled intervention argues against crediting this difference
for CUDA's speed. Giving Mojo independent per-cell `x`, `h`, error, and
rejection state reduced its median attempts from 1,687 to 1,543—matching
CUDA's median—but changed the isolated mean from 36.6046 s to 37.0186 s, a
1.13% regression. Its maximum attempt count fell only from 9,262 to 9,118,
which matters more because the CTA still runs until its hardest cell finishes.
The experiment also raised registers from 238 to 240 and added control-flow
divergence while leaving occupancy and barriers unchanged.

CUDA's controller and thread-per-cell mapping are bundled together, so the
saved CUDA counters cannot identify the controller's independent causal
effect. CUDA warps still execute in SIMT: when cell histories diverge, a warp
continues until its slowest active thread finishes. Numerical substep
independence may therefore be neutral or harmful even in CUDA. At most, the
lower attempt count is a modest logical-work advantage whose runtime value is
unproven. It should not be treated as one of the main explanations for the
factor-of-two result.

## Finding: CUDA's Spill Penalty Is Large but Not Dominant

Confidence: **high for the traffic and stall measurements; the isolated
runtime cost is not directly measured**.

The PTXAS report should not be read as "thousands of physical registers are
spilled at once." CUDA still uses 255 hardware registers per thread. Its
approximately 6.3 KiB stack frame and spill load/store figures describe
compiler-managed per-thread local storage and generated accesses, not a count
of simultaneously live physical registers.

The full-grid profile shows that the spill traffic is neither hypothetical nor
well coalesced. In launch 499, CUDA executes 36.43 million local loads and
31.20 million local stores; 15.68 million SASS instructions are specifically
identified by NCU as register spilling. Those accesses generate 287.4 million
load sectors and 246.7 million store sectors. NCU reports only 54.66% and
47.65% L1 hit rates, and flags an average local-sector utilization of only 1
out of 32. The earlier claim that naturally coalesced local accesses might
make the spills cheap is not supported for this kernel.

The cost is visible in CUDA's 5.155 long-scoreboard cycles per issued
instruction, 26.27% of its issue interval. Nevertheless, the launch moves
3.21 GB from DRAM and writes 4.59 GB while sustaining 492.70 GB/s, only 14.02%
of peak memory throughput. NCU's local-access rule estimates about 9.94% local
speedup from perfecting the access pattern. That heuristic is not a controlled
spill-removal experiment, so it suggests rather than proves the isolated cost.
The spill path is a serious secondary cost, not a demonstrated factor-of-two
ceiling.

CUDA remains 1.89 times faster in this same launch because it executes 3.05
times fewer warp instructions and about 2.47 times fewer predicated-on thread
instructions. The stack allocation is also not proof that every byte is hot
on every attempt. Spills slow CUDA down; the measured work advantage is simply
larger.

## Secondary Hypothesis: Shared-Memory Materialization Is More Expensive Than
It Looks

Confidence: **high that the cost exists, medium that it is a major fraction of
the 2-times gap**.

Mojo externalizes the generated DAG into 531 shared `Float64` scratch values
per cell, plus the Jacobian, RHS, inputs, and exchange arena. This gives exact
cross-slice CSE and zero spills, but each producer/consumer boundary creates
stores, loads, address arithmetic, and ordering constraints.

The production launch-499 profile reports:

- 117.23 million shared loads and 58.49 million shared stores.
- 2.1-way conflicts for shared loads, adding 64.22 million conflicts.
- 3.0-way conflicts for shared stores, adding 55.55 million conflicts.
- 424.27 million shared wavefronts versus 276.96 million ideal wavefronts:
  147.31 million excessive wavefronts, or 35% of the actual total.
- An NCU source-counter estimate of 26.5% local speedup if those excessive
  wavefronts could be eliminated. This is an optimization heuristic, not a
  measured end-to-end speedup.

Shared memory is lower latency than uncached local memory, but it is not free.
Here the shared-memory and register allocations each independently limit the
SM to one CTA. Materialization also contributes to the measured instruction
gap and requires the barriers that dominate Mojo's stall breakdown. The full
profile therefore strengthens this hypothesis, although it still cannot
separate DAG exchange from cooperative LU and controller overhead.

## Secondary Hypothesis: Cooperative LU Is Too Fine-Grained for a 15-by-15
System

Confidence: **medium-high**.

CUDA performs each cell's pivoted 15-by-15 LU and three solves serially within
one thread. This spills the matrix, but it needs no interthread coordination.

Mojo assigns an eight-lane subgroup to each cell and four cell groups to each
warp. Factorization and solves repeatedly synchronize the whole physical warp
after pivot selection, swaps, scaling, elimination, and forward/backward
substitution. As the triangular loops shrink, fewer subgroup lanes have useful
work, but all lanes still reach each synchronization point. Four cells in the
same physical warp are also coupled by each `syncwarp` even if their pivot or
control paths differ.

For a matrix this small, the available parallel arithmetic per synchronization
is modest. CUDA's serial LU may therefore have worse single-cell latency but
better throughput across a warp of 32 independent cells.

## Mixed Finding: Compiler Scheduling Freedom

Confidence: **medium-low as a net explanation**.

The CUDA kernel presents a monolithic per-thread computation. NVCC/PTXAS may
schedule independent arithmetic and local loads across a large instruction
window, and spilling can shorten some otherwise extreme live ranges.

The Mojo implementation divides the graph into generated functions and
barrier-delimited waves. Shared-memory handoffs and barriers prevent motion
across those boundaries. In addition, `portable_mul` uses inline PTX to force
separately rounded FP64 multiplication. Although marked side-effect-free, its
explicit input/output constraints can restrict optimization more than CUDA's
ordinary multiplication compiled with `--fmad=false`.

The direct comparison is mixed. Mojo spends 62.34% of its issue interval at
barriers, which confirms that phase boundaries constrain motion and overlap.
But CUDA spends 49.43% with no instruction ready to fetch, consistent with
instruction-cache or fetch pressure from its very large monolithic body, and
its overall cycles per issued instruction are slightly worse than Mojo's.
CUDA does not demonstrate more scheduling headroom at the aggregate scheduler
level. Its advantage is that this imperfect schedule processes a much smaller
dynamic instruction stream.

## Factors Unlikely to Explain the Gap

- **Host or launch overhead:** chemistry kernels account for 99.8% of GPU time
  in both saved runs; the kernel-time ratio is larger than the application
  ratio.
- **Preparation kernels:** 63.7 ms for Mojo versus 34.6 ms for CUDA over a
  multi-second run is negligible relative to the gap.
- **Missing FMA alone:** Nsight Compute estimates only about 1.3% local upside
  from converting the remaining Mojo FP64 multiply/add pairs.
- **Occupancy or aggregate issue eligibility:** both kernels have 12.5%
  theoretical occupancy, about two active warps per scheduler, and about 90%
  no-eligible cycles. CUDA is marginally worse on the latter measure.
- **Raw DRAM bandwidth:** Mojo reaches only 3.53% and CUDA only 14.02% of peak
  memory throughput in launch 499. CUDA suffers memory latency and traffic,
  but neither kernel demonstrates an external-bandwidth ceiling.
- **Independent numerical substepping by itself:** the controlled Mojo variant
  reduced attempts but regressed by 1.13%. The CUDA profile's 99.65% branch
  efficiency shows little within-warp divergence, but does not establish a
  positive causal speedup from the controller policy.

## Combined Hypothesis

The measured result now supports this ranking:

1. **Dynamic work is the main empirical separator.** Mojo executes 3.05 times
   as many warp instructions and about 2.47 times as many predicated-on thread
   instructions in both the middle and late launches.
2. **Algorithmic reuse contributes to that gap.** CUDA avoids roughly 22–29%
   of Mojo's Jacobian evaluations by preserving the raw Jacobian across
   retries, and its counters show fewer RHS calls and attempts.
3. **Mojo's execution topology adds overhead.** Shared DAG materialization,
   147.31 million excessive shared wavefronts, cooperative 15-by-15 LU,
   role-specialized partial warps, reductions, and phase barriers plausibly
   account for much of the remaining non-FP64 instruction difference.
4. **Barriers hurt Mojo but do not create a scheduler-eligibility contrast.**
   Mojo spends about 62% of issue latency at barriers; CUDA has zero barrier
   stalls but is equally scheduler-starved by instruction fetch and memory
   dependencies. CUDA wins by traversing a shorter instruction stream.
5. **CUDA's spills are a substantial tax, not the dominant term.** They cause
   hundreds of millions of local-memory sectors and 26% long-scoreboard stall
   time, yet use only 14% of peak memory throughput and do not erase CUDA's
   2.5-times effective instruction-work advantage.
6. **Independent substepping is not credited as a speed mechanism.** CUDA
   records fewer attempts, but the controlled Mojo intervention regressed.

The central lesson is more precise than the original hypothesis: minimizing
spills does not minimize dynamic work. The structured kernel exchanges a huge
local-memory problem for shared-memory materialization, synchronization, bank
conflicts, partial-warp execution, and extra control instructions. On this
workload, CUDA's much shorter, nearly full-warp dynamic instruction stream is
worth more than the spill traffic it incurs.

## Evidence Sources

- `benchmark_runs/fused-12752255/build-cuda.log`
- `benchmark_runs/fused-12752255/ncu-structured-details.txt`
- `benchmark_runs/fused-12752255/nsys-cuda-stats.txt`
- `benchmark_runs/fused-12752255/nsys-structured-stats.txt`
- `benchmark_runs/fused-12752255/run1-cuda.log`
- `benchmark_runs/fused-12752255/run1-structured.log`
- `benchmark_runs/ncu-grid64-12754915/environment.txt`
- `benchmark_runs/ncu-grid64-12754915/{structured,cuda}-{early,middle,late}-details.txt`
- `benchmark_runs/ncu-grid64-12754915/{structured,cuda}-{early,middle,late}-raw.csv`
- `benchmark_runs/ncu-grid64-12754915/{structured,cuda}-{early,middle,late}.ncu-rep`
- `benchmarks/ncu_grid64_compare.slurm`
- CUDA controller and LU: `reproducer.cpp:97`, `reproducer.cpp:7205`, and
  `reproducer.cpp:7734`
- Structured scheduling and LU: `structured_ops.mojo:59`,
  `structured_trial.mojo:195`, `structured_trial.mojo:540`, and
  `structured_trial.mojo:902`

# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: Shared grid preparation and data-parallel GPU chemistry kernels.

import reproducer
from layout import TensorLayout, TileTensor, row_major, stack_allocation
from std.gpu import barrier, block_idx, lane_id
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp
from std.math import abs, exp, isfinite, log, max, min


comptime StateFields = 19
comptime StatsFields = 7
comptime RhoField = 0
comptime TemperatureField = 1
comptime EnergyField = 2
comptime SpeciesField = 3
comptime TimeField = 17
comptime DriverField = 18


def gpu_cbrt_nonnegative(value: Float64) -> Float64:
    if value == 0.0:
        return 0.0
    return exp(log(value) / 3.0)


def load_f64[LayoutType: TensorLayout](
    tensor: TileTensor[DType.float64, LayoutType, MutAnyOrigin],
    field: Int,
    cell: Int,
) -> Float64:
    comptime assert tensor.flat_rank == 2
    return rebind[Scalar[DType.float64]](tensor[field, cell])


def store_f64[LayoutType: TensorLayout](
    tensor: TileTensor[DType.float64, LayoutType, MutAnyOrigin],
    field: Int,
    cell: Int,
    value: Float64,
):
    comptime assert tensor.flat_rank == 2
    tensor[field, cell] = rebind[tensor.ElementType](value)


def load_i32[LayoutType: TensorLayout](
    tensor: TileTensor[DType.int32, LayoutType, MutAnyOrigin], cell: Int
) -> Int:
    comptime assert tensor.flat_rank == 1
    return Int(rebind[Scalar[DType.int32]](tensor[cell]))


def store_i32[LayoutType: TensorLayout](
    tensor: TileTensor[DType.int32, LayoutType, MutAnyOrigin],
    cell: Int,
    value: Int,
):
    comptime assert tensor.flat_rank == 1
    tensor[cell] = rebind[tensor.ElementType](Int32(value))


def load_u64[LayoutType: TensorLayout](
    tensor: TileTensor[DType.uint64, LayoutType, MutAnyOrigin],
    field: Int,
    cell: Int,
) -> Int:
    comptime assert tensor.flat_rank == 2
    return Int(rebind[Scalar[DType.uint64]](tensor[field, cell]))


def store_u64[LayoutType: TensorLayout](
    tensor: TileTensor[DType.uint64, LayoutType, MutAnyOrigin],
    field: Int,
    cell: Int,
    value: Int,
):
    comptime assert tensor.flat_rank == 2
    tensor[field, cell] = rebind[tensor.ElementType](UInt64(value))


def load_collapse[
    StateLayout: TensorLayout,
    StepLayout: TensorLayout,
    StatsLayout: TensorLayout,
](
    state_tensor: TileTensor[DType.float64, StateLayout, MutAnyOrigin],
    steps: TileTensor[DType.int32, StepLayout, MutAnyOrigin],
    stats: TileTensor[DType.uint64, StatsLayout, MutAnyOrigin],
    cell: Int,
) -> reproducer.CollapseState:
    var state = reproducer.CollapseState()
    state.current.rho = load_f64(state_tensor, RhoField, cell)
    state.current.T = load_f64(state_tensor, TemperatureField, cell)
    state.current.e = load_f64(state_tensor, EnergyField, cell)
    for n in range(reproducer.NumSpec):
        state.current.xn[n] = load_f64(state_tensor, SpeciesField + n, cell)
    state.time = load_f64(state_tensor, TimeField, cell)
    state.density_driver = load_f64(state_tensor, DriverField, cell)
    state.completed_steps = load_i32(steps, cell)
    state.stats.internal_steps = load_u64(stats, 0, cell)
    state.stats.rhs_calls = load_u64(stats, 1, cell)
    state.stats.jacobian_calls = load_u64(stats, 2, cell)
    state.stats.decompositions = load_u64(stats, 3, cell)
    state.stats.linear_solves = load_u64(stats, 4, cell)
    state.stats.accepted_steps = load_u64(stats, 5, cell)
    state.stats.rejected_steps = load_u64(stats, 6, cell)
    return state^


def store_collapse[
    StateLayout: TensorLayout,
    StepLayout: TensorLayout,
    StatsLayout: TensorLayout,
](
    state_tensor: TileTensor[DType.float64, StateLayout, MutAnyOrigin],
    steps: TileTensor[DType.int32, StepLayout, MutAnyOrigin],
    stats: TileTensor[DType.uint64, StatsLayout, MutAnyOrigin],
    cell: Int,
    state: reproducer.CollapseState,
):
    store_f64(state_tensor, RhoField, cell, state.current.rho)
    store_f64(state_tensor, TemperatureField, cell, state.current.T)
    store_f64(state_tensor, EnergyField, cell, state.current.e)
    for n in range(reproducer.NumSpec):
        store_f64(state_tensor, SpeciesField + n, cell, state.current.xn[n])
    store_f64(state_tensor, TimeField, cell, state.time)
    store_f64(state_tensor, DriverField, cell, state.density_driver)
    store_i32(steps, cell, state.completed_steps)
    store_u64(stats, 0, cell, state.stats.internal_steps)
    store_u64(stats, 1, cell, state.stats.rhs_calls)
    store_u64(stats, 2, cell, state.stats.jacobian_calls)
    store_u64(stats, 3, cell, state.stats.decompositions)
    store_u64(stats, 4, cell, state.stats.linear_solves)
    store_u64(stats, 5, cell, state.stats.accepted_steps)
    store_u64(stats, 6, cell, state.stats.rejected_steps)


def prepare_grid_timestep_kernel[
    StateLayout: TensorLayout,
    StepLayout: TensorLayout,
    StatsLayout: TensorLayout,
    MinLayout: TensorLayout,
    IdLayout: TensorLayout,
    FailureLayout: TensorLayout,
](
    state_tensor: TileTensor[DType.float64, StateLayout, MutAnyOrigin],
    steps: TileTensor[DType.int32, StepLayout, MutAnyOrigin],
    stats: TileTensor[DType.uint64, StatsLayout, MutAnyOrigin],
    cta_dt_min: TileTensor[DType.float64, MinLayout, MutAnyOrigin],
    cta_min_cell: TileTensor[DType.int32, IdLayout, MutAnyOrigin],
    failure: TileTensor[DType.int32, FailureLayout, MutAnyOrigin],
    num_cells: Int,
    completed_global_steps: Int,
    grid_time: Float64,
    step: Int,
    perturb_value: Int32,
):
    comptime assert cta_dt_min.flat_rank == 1
    comptime assert cta_min_cell.flat_rank == 1
    comptime assert failure.flat_rank == 1
    var dt_shared = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[reproducer.TileCells]())
    var lane = lane_id()
    var cell = block_idx.x * reproducer.TileCells + lane
    var candidate = Float64.MAX
    var candidate_cell = Int.MAX
    if lane < reproducer.TileCells and cell < num_cells:
        var collapse = load_collapse(state_tensor, steps, stats, cell)
        if collapse.completed_steps == completed_global_steps:
            if collapse.time != grid_time:
                failure[0] = Int32(reproducer.BadInputs)
            else:
                reproducer.apply_perturbation(
                    collapse, cell, step, perturb_value != 0
                )
                var dt = reproducer.collapse_timestep(collapse)
                if dt <= 0.0 or not reproducer.valid_positive(collapse.density_driver):
                    failure[0] = Int32(reproducer.BadInputs)
                elif dt < 10.0:
                    collapse.density_driver += (
                        dt * collapse.density_driver / (dt / reproducer.TffReduc)
                    )
                else:
                    candidate = dt
                    candidate_cell = cell
                # Preparation owns every mutation it makes, including perturbation.
                store_f64(state_tensor, RhoField, cell, collapse.current.rho)
                store_f64(
                    state_tensor,
                    TemperatureField,
                    cell,
                    collapse.current.T,
                )
                store_f64(state_tensor, EnergyField, cell, collapse.current.e)
                for n in range(reproducer.NumSpec):
                    store_f64(
                        state_tensor,
                        SpeciesField + n,
                        cell,
                        collapse.current.xn[n],
                    )
                store_f64(
                    state_tensor,
                    DriverField,
                    cell,
                    collapse.density_driver,
                )
    if lane < reproducer.TileCells:
        dt_shared[lane] = candidate
    barrier()
    if lane == 0:
        var minimum = Float64.MAX
        var minimum_cell = Int.MAX
        for logical_lane in range(reproducer.TileCells):
            var lane_dt = rebind[Scalar[DType.float64]](
                dt_shared[logical_lane]
            )
            var lane_cell = block_idx.x * reproducer.TileCells + logical_lane
            if lane_dt < minimum or (
                lane_dt == minimum and lane_cell < minimum_cell
            ):
                minimum = lane_dt
                minimum_cell = lane_cell
        cta_dt_min[block_idx.x] = minimum
        cta_min_cell[block_idx.x] = Int32(minimum_cell)


def chemistry_kernel[
    StateLayout: TensorLayout,
    StepLayout: TensorLayout,
    StatsLayout: TensorLayout,
    IdLayout: TensorLayout,
    FailureLayout: TensorLayout,
](
    state_tensor: TileTensor[DType.float64, StateLayout, MutAnyOrigin],
    steps: TileTensor[DType.int32, StepLayout, MutAnyOrigin],
    stats: TileTensor[DType.uint64, StatsLayout, MutAnyOrigin],
    integrated_by_tile: TileTensor[DType.int32, IdLayout, MutAnyOrigin],
    failure: TileTensor[DType.int32, FailureLayout, MutAnyOrigin],
    num_cells: Int,
    completed_global_steps: Int,
    grid_time: Float64,
    next_grid_time: Float64,
    dt_grid: Float64,
):
    comptime assert integrated_by_tile.flat_rank == 1
    comptime assert failure.flat_rank == 1
    var trial_errors = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[reproducer.TileCells]())
    var shared_error = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[1]())
    var lane = lane_id()
    var cell = block_idx.x * reproducer.TileCells + lane
    var valid = lane < reproducer.TileCells and cell < num_cells
    var collapse = reproducer.CollapseState()
    if valid:
        collapse = load_collapse(state_tensor, steps, stats, cell)

    var participant = False
    if valid and collapse.completed_steps == completed_global_steps:
        if collapse.time != grid_time:
            failure[0] = Int32(reproducer.BadInputs)
        else:
            var local_dt = reproducer.collapse_timestep(collapse)
            if local_dt >= 10.0:
                var old_density = collapse.density_driver
                collapse.density_driver += reproducer.portable_mul(
                    dt_grid,
                    collapse.density_driver,
                ) / (local_dt / reproducer.TffReduc)
                if (
                    reproducer.valid_positive(collapse.density_driver)
                    and collapse.density_driver <= 2.0e-6
                ):
                    var ratio = collapse.density_driver / old_density
                    for n in range(reproducer.NumSpec):
                        collapse.current.xn[n] *= ratio
                    collapse.current.rho *= ratio
                    reproducer.eos_rt(collapse.current)
                    participant = True

    var participant_count = Int(warp.sum(Int32(participant)))
    var candidate = reproducer.BurnState()
    comptime uround = 1.0e-16
    comptime fac_min = 0.2
    comptime fac_max = 6.0
    comptime safe = 0.9
    comptime max_steps = 10000000
    var h = dt_grid
    var x = 0.0
    var reject = False
    var nsing = 0
    var hacc = 0.0
    var erracc = 1.0
    var n_step = 0
    var n_accept = 0
    var status = reproducer.Success

    while True:
        if (
            participant_count == 0
            or x >= dt_grid
            or status != reproducer.Success
        ):
            break
        if n_step > max_steps:
            status = reproducer.TooManySteps
            continue
        if 0.1 * abs(h) <= abs(x) * uround:
            status = reproducer.DtUnderflow
            continue
        var final_trial = False
        if x + reproducer.portable_mul(h, 1.0001) >= dt_grid:
            h = dt_grid - x
            final_trial = True

        var trial_status = reproducer.Success
        var cell_error = 0.0
        if participant:
            var trial = reproducer.ros2s_trial(
                collapse.current, h, x, candidate, collapse.stats
            )
            trial_status = trial.status
            cell_error = trial.error
        var singular = Int(
            warp.max(Int32(trial_status == reproducer.LuDecompositionError))
        )
        if singular != 0:
            nsing += 1
            if nsing >= 5:
                status = reproducer.LuDecompositionError
            else:
                h *= 0.5
                reject = True
            continue

        var nonfinite = Int(
            warp.max(Int32(participant and not isfinite(cell_error)))
        )
        if nonfinite != 0:
            cell_error = Float64.MAX
        if lane < reproducer.TileCells:
            trial_errors[lane] = cell_error
        barrier()
        if lane == 0:
            var tile_max = 0.0
            for logical_lane in range(reproducer.TileCells):
                tile_max = max(
                    tile_max,
                    rebind[Scalar[DType.float64]](
                        trial_errors[logical_lane]
                    ),
                )
            shared_error[0] = tile_max
        barrier()
        var err_tile = rebind[Scalar[DType.float64]](shared_error[0])
        n_step += 1
        var fac_step = max(
            1.0 / fac_max,
            min(1.0 / fac_min, gpu_cbrt_nonnegative(err_tile) / safe),
        )
        var hnew = h / fac_step
        if err_tile <= 1.0:
            n_accept += 1
            if n_accept > 1:
                var facgus = max(
                    1.0 / fac_max,
                    min(
                        1.0 / fac_min,
                        (hacc / h)
                        * gpu_cbrt_nonnegative(
                            (err_tile * err_tile) / erracc
                        )
                        / safe,
                    ),
                )
                hnew = h / max(fac_step, facgus)
            hacc = h
            erracc = max(1.0e-2, err_tile)
            if participant:
                for n in range(reproducer.NumSpec):
                    collapse.current.xn[n] = candidate.xn[n]
                collapse.current.e = candidate.e
                collapse.stats.accepted_steps += 1
            x = dt_grid if final_trial else x + h
            hnew = min(abs(hnew), dt_grid)
            if reject:
                hnew = min(hnew, abs(h))
            reject = False
            h = hnew
        else:
            reject = True
            h = hnew
            if participant and n_accept >= 1:
                collapse.stats.rejected_steps += 1

    if status == reproducer.Success and participant:
        reproducer.floor_and_normalize_number_densities(collapse.current)
        reproducer.balance_charge(collapse.current)
        reproducer.floor_and_normalize_number_densities(collapse.current)
        reproducer.eos_re(collapse.current)
        collapse.time = next_grid_time
        collapse.completed_steps = completed_global_steps + 1
    if valid:
        store_collapse(state_tensor, steps, stats, cell, collapse)
    if lane == 0:
        integrated_by_tile[block_idx.x] = Int32(
            participant_count if status == reproducer.Success else 0
        )
        if status != reproducer.Success:
            failure[0] = Int32(status)

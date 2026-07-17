# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: Hopper single-batch structured ROS2S kernel with transactional output.

import reproducer
import reproducer_gpu
import structured_ops
import warp_lu
from layout import TensorLayout, TileTensor, row_major, stack_allocation
from std.gpu import WARP_SIZE, barrier, block_idx, thread_idx
from std.gpu.memory import AddressSpace, external_memory
from std.math import abs, exp, isfinite, log, max, min


def gpu_cbrt_nonnegative(value: Float64) -> Float64:
    if value == 0.0:
        return 0.0
    return exp(log(value) / 3.0)


def factorize_columns[
    PivotLayout: TensorLayout,
    ExchangeLayout: TensorLayout,
    InfoLayout: TensorLayout,
](
    mut op: warp_lu.WarpLuOp,
    owned: Bool,
    warp_id: Int,
    lane: Int,
    pivots: TileTensor[
        DType.int32,
        PivotLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    exchange: TileTensor[
        DType.float64,
        ExchangeLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    infos: TileTensor[
        DType.int32,
        InfoLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
):
    comptime assert pivots.flat_rank == 2
    comptime assert exchange.flat_rank == 2
    comptime assert infos.flat_rank == 1
    if owned and lane == 0:
        infos[warp_id] = Int32(0)
    barrier()
    for k in range(reproducer.Neqs - 1):
        if owned and lane == k:
            op.select_pivot(k)
            pivots[warp_id, k] = Int32(op.pivot_row)
        barrier()
        var pivot = k
        if owned:
            pivot = Int(rebind[Scalar[DType.int32]](pivots[warp_id, k]))
        if owned and lane >= k and lane < reproducer.Neqs:
            op.swap_rows(k, pivot)
        barrier()
        if owned and lane == k:
            if op.column[k] == 0.0:
                infos[warp_id] = Int32(k + 1)
            else:
                var scale = -1.0 / op.column[k]
                for row in range(k + 1, reproducer.Neqs):
                    op.column[row] *= scale
        barrier()
        for row in range(k + 1, reproducer.Neqs):
            if owned and lane == k:
                exchange[warp_id, 0] = (
                    op.column[row] if op.column[k] != 0.0 else 0.0
                )
            barrier()
            if owned and lane > k and lane < reproducer.Neqs:
                op.column[row] += reproducer.portable_mul(
                    op.column[k],
                    rebind[Scalar[DType.float64]](exchange[warp_id, 0]),
                )
            barrier()
    if (
        owned
        and lane == reproducer.Neqs - 1
        and op.column[reproducer.Neqs - 1] == 0.0
    ):
        infos[warp_id] = Int32(reproducer.Neqs)
    barrier()


def solve_columns[
    PivotLayout: TensorLayout,
    ExchangeLayout: TensorLayout,
    InfoLayout: TensorLayout,
](
    mut op: warp_lu.WarpLuOp,
    owned: Bool,
    warp_id: Int,
    lane: Int,
    pivots: TileTensor[
        DType.int32,
        PivotLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    exchange: TileTensor[
        DType.float64,
        ExchangeLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    infos: TileTensor[
        DType.int32,
        InfoLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
):
    comptime assert pivots.flat_rank == 2
    comptime assert exchange.flat_rank == 2
    comptime assert infos.flat_rank == 1
    var nonsingular = owned
    if owned:
        nonsingular = rebind[Scalar[DType.int32]](infos[warp_id]) == 0
    for k in range(reproducer.Neqs - 1):
        var pivot = k
        if owned:
            pivot = Int(rebind[Scalar[DType.int32]](pivots[warp_id, k]))
        if nonsingular and lane == k:
            exchange[warp_id, 0] = op.rhs
        if nonsingular and lane == pivot:
            exchange[warp_id, 1] = op.rhs
        barrier()
        if nonsingular and lane < reproducer.Neqs:
            var x_k = rebind[Scalar[DType.float64]](exchange[warp_id, 0])
            var x_pivot = rebind[Scalar[DType.float64]](
                exchange[warp_id, 1]
            )
            if lane == pivot:
                op.rhs = x_k
            if lane == k:
                op.rhs = x_pivot
        barrier()
        if nonsingular and lane == k:
            exchange[warp_id, 0] = op.rhs
        barrier()
        var t = 0.0
        if nonsingular:
            t = rebind[Scalar[DType.float64]](exchange[warp_id, 0])
        for row in range(k + 1, reproducer.Neqs):
            if nonsingular and lane == k:
                exchange[warp_id, 1] = op.column[row]
            barrier()
            if nonsingular and lane == row:
                op.rhs += reproducer.portable_mul(
                    t,
                    rebind[Scalar[DType.float64]](exchange[warp_id, 1]),
                )
            barrier()
    for reverse_k in range(reproducer.Neqs):
        var k = reproducer.Neqs - 1 - reverse_k
        if nonsingular and lane == k:
            op.rhs /= op.column[k]
            exchange[warp_id, 0] = op.rhs
        barrier()
        var negative_x = 0.0
        if nonsingular:
            negative_x = -rebind[Scalar[DType.float64]](
                exchange[warp_id, 0]
            )
        for row in range(k):
            if nonsingular and lane == k:
                exchange[warp_id, 1] = op.column[row]
            barrier()
            if nonsingular and lane == row:
                op.rhs += reproducer.portable_mul(
                    negative_x,
                    rebind[Scalar[DType.float64]](exchange[warp_id, 1]),
                )
            barrier()


def factorize_shared[
    PivotLayout: TensorLayout,
    InfoLayout: TensorLayout,
](
    jacobian: UnsafePointer[
        Scalar[DType.float64],
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    participating: Bool,
    cell: Int,
    local_lane: Int,
    pivots: TileTensor[
        DType.int32,
        PivotLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    infos: TileTensor[
        DType.int32,
        InfoLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
):
    comptime assert pivots.flat_rank == 2
    comptime assert infos.flat_rank == 1
    comptime matrix_values = reproducer.Neqs * reproducer.Neqs
    if local_lane == 0:
        infos[cell] = Int32(0)
    barrier()
    for k in range(reproducer.Neqs - 1):
        if participating and local_lane == k % structured_ops.LuGroupWidth:
            var pivot = k
            var maximum = abs(
                jacobian[cell * matrix_values + k * reproducer.Neqs + k]
            )
            for row in range(k + 1, reproducer.Neqs):
                var value = abs(
                    jacobian[
                        cell * matrix_values + row * reproducer.Neqs + k
                    ]
                )
                if value > maximum:
                    maximum = value
                    pivot = row
            pivots[cell, k] = Int32(pivot)
        barrier()
        var pivot = k
        if participating:
            pivot = Int(rebind[Scalar[DType.int32]](pivots[cell, k]))
        if participating:
            for column in range(
                local_lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                if column >= k and pivot != k:
                    var top = cell * matrix_values + k * reproducer.Neqs + column
                    var bottom = (
                        cell * matrix_values
                        + pivot * reproducer.Neqs
                        + column
                    )
                    var temporary = jacobian[bottom]
                    jacobian[bottom] = jacobian[top]
                    jacobian[top] = temporary
        barrier()
        if participating and local_lane == k % structured_ops.LuGroupWidth:
            var diagonal_index = (
                cell * matrix_values + k * reproducer.Neqs + k
            )
            var diagonal = jacobian[diagonal_index]
            if diagonal == 0.0:
                infos[cell] = Int32(k + 1)
            else:
                var scale = -1.0 / diagonal
                for row in range(k + 1, reproducer.Neqs):
                    var index = (
                        cell * matrix_values + row * reproducer.Neqs + k
                    )
                    jacobian[index] *= scale
        barrier()
        if participating:
            for column in range(
                local_lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                if column > k:
                    var upper = jacobian[
                        cell * matrix_values + k * reproducer.Neqs + column
                    ]
                    for row in range(k + 1, reproducer.Neqs):
                        var index = (
                            cell * matrix_values
                            + row * reproducer.Neqs
                            + column
                        )
                        var multiplier = jacobian[
                            cell * matrix_values
                            + row * reproducer.Neqs
                            + k
                        ]
                        jacobian[index] += reproducer.portable_mul(
                            upper, multiplier
                        )
        barrier()
    if participating and local_lane == 0:
        if jacobian[cell * matrix_values + matrix_values - 1] == 0.0:
            infos[cell] = Int32(reproducer.Neqs)
    barrier()


def solve_shared[
    PivotLayout: TensorLayout,
    InfoLayout: TensorLayout,
](
    jacobian: UnsafePointer[
        Scalar[DType.float64],
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    rhs: UnsafePointer[
        Scalar[DType.float64],
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    participating: Bool,
    cell: Int,
    local_lane: Int,
    pivots: TileTensor[
        DType.int32,
        PivotLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    infos: TileTensor[
        DType.int32,
        InfoLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
):
    comptime assert pivots.flat_rank == 2
    comptime assert infos.flat_rank == 1
    comptime matrix_values = reproducer.Neqs * reproducer.Neqs
    var nonsingular = participating
    if participating:
        nonsingular = rebind[Scalar[DType.int32]](infos[cell]) == 0
    for k in range(reproducer.Neqs - 1):
        if nonsingular and local_lane == 0:
            var pivot = Int(
                rebind[Scalar[DType.int32]](pivots[cell, k])
            )
            if pivot != k:
                var temporary = rhs[cell * reproducer.Neqs + pivot]
                rhs[cell * reproducer.Neqs + pivot] = (
                    rhs[cell * reproducer.Neqs + k]
                )
                rhs[cell * reproducer.Neqs + k] = temporary
        barrier()
        if nonsingular:
            var value = rhs[cell * reproducer.Neqs + k]
            for row in range(
                local_lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                if row > k:
                    rhs[cell * reproducer.Neqs + row] = (
                        rhs[cell * reproducer.Neqs + row]
                        + reproducer.portable_mul(
                            value,
                            jacobian[
                                cell * matrix_values
                                + row * reproducer.Neqs
                                + k
                            ],
                        )
                    )
        barrier()
    for reverse_k in range(reproducer.Neqs):
        var k = reproducer.Neqs - 1 - reverse_k
        if nonsingular and local_lane == k % structured_ops.LuGroupWidth:
            rhs[cell * reproducer.Neqs + k] = (
                rhs[cell * reproducer.Neqs + k]
                / jacobian[cell * matrix_values + k * reproducer.Neqs + k]
            )
        barrier()
        if nonsingular:
            var negative_x = -rhs[cell * reproducer.Neqs + k]
            for row in range(
                local_lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                if row < k:
                    rhs[cell * reproducer.Neqs + row] = (
                        rhs[cell * reproducer.Neqs + row]
                        + reproducer.portable_mul(
                            negative_x,
                            jacobian[
                                cell * matrix_values
                                + row * reproducer.Neqs
                                + k
                            ],
                        )
                    )
        barrier()


def evaluate_jacobian_batch[
    BaseLayout: TensorLayout,
](
    base: TileTensor[
        DType.float64,
        BaseLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    jacobian: UnsafePointer[
        Scalar[DType.float64],
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    warp_id: Int,
    lane: Int,
    active_cells: Int,
):
    comptime assert base.flat_rank == 2
    var tile_cell = lane
    if (
        warp_id < structured_ops.DagWarps and tile_cell < active_cells
    ):
        var y = reproducer.VecTensor.stack_allocation()
        for component in range(reproducer.Neqs):
            reproducer.vset(
                y,
                component,
                rebind[Scalar[DType.float64]](base[tile_cell, component]),
            )
        var state = reproducer.burn_state_from_y(y)
        var X = reproducer.SpeciesTensor.stack_allocation()
        for species in range(reproducer.NumSpec):
            reproducer.vset(X, species, state.xn[species])
        var local = reproducer.MatTensor.stack_allocation()
        structured_ops.dispatch_jac_slice(
            warp_id, state, local, X, reproducer.redshift()
        )
        var first_row = warp_id * 2
        var last_row = min(first_row + 2, reproducer.Neqs)
        for row in range(first_row, last_row):
            for column in range(reproducer.Neqs):
                var index = (
                    tile_cell * reproducer.Neqs * reproducer.Neqs
                    + row * reproducer.Neqs
                    + column
                )
                jacobian[index] = reproducer.mget(local, row, column)


def evaluate_rhs_batch[
    InputLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    inputs: TileTensor[
        DType.float64,
        InputLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    outputs: TileTensor[
        DType.float64,
        OutputLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    warp_id: Int,
    lane: Int,
    active_cells: Int,
):
    comptime assert inputs.flat_rank == 2
    comptime assert outputs.flat_rank == 2
    var tile_cell = lane
    if (
        warp_id < structured_ops.DagWarps and tile_cell < active_cells
    ):
        var y = reproducer.VecTensor.stack_allocation()
        for component in range(reproducer.Neqs):
            reproducer.vset(
                y,
                component,
                rebind[Scalar[DType.float64]](inputs[tile_cell, component]),
            )
        var state = reproducer.burn_state_from_y(y)
        var X = reproducer.SpeciesTensor.stack_allocation()
        for species in range(reproducer.NumSpec):
            reproducer.vset(X, species, state.xn[species])
        var local = reproducer.VecTensor.stack_allocation()
        structured_ops.dispatch_rhs_slice(
            warp_id, state, local, X, reproducer.redshift()
        )
        outputs[lane, warp_id] = reproducer.vget(local, warp_id)
        var second = warp_id + structured_ops.DagWarps
        if second < reproducer.NumSpec:
            outputs[lane, second] = reproducer.vget(local, second)
        if warp_id == 6:
            outputs[lane, reproducer.NetIenuc] = reproducer.vget(
                local, reproducer.NetIenuc
            )


def prepare_dag_inputs[BaseLayout: TensorLayout](
    base: TileTensor[
        DType.float64,
        BaseLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    inputs: structured_ops.SharedPointer,
    tid: Int,
    active_cells: Int,
):
    comptime assert base.flat_rank == 2
    if tid < active_cells:
        var rhotot = 0.0
        for species in range(reproducer.NumSpec):
            var value = max(
                rebind[Scalar[DType.float64]](base[tid, species]),
                reproducer.small_number_density_floor(),
            )
            inputs[tid * reproducer.Neqs + species] = value
            rhotot += reproducer.portable_mul(
                value, reproducer.species_mass(species)
            )
        var sum_abarinv = 0.0
        var sum_gammasinv = 0.0
        for species in range(reproducer.NumSpec):
            var value = inputs[tid * reproducer.Neqs + species]
            sum_abarinv += value
            sum_gammasinv += reproducer.portable_mul(
                reproducer.portable_mul(value, reproducer.MP) / rhotot,
                1.0 / (reproducer.species_gamma(species) - 1.0),
            )
        sum_abarinv *= reproducer.MP / rhotot
        sum_gammasinv /= sum_abarinv
        var energy = rebind[Scalar[DType.float64]](
            base[tid, reproducer.NetIenuc]
        )
        inputs[tid * reproducer.Neqs + reproducer.NetIenuc] = energy / (
            reproducer.portable_mul(
                reproducer.portable_mul(
                    sum_gammasinv, reproducer.NA * reproducer.KB
                ),
                sum_abarinv,
            )
        )
    barrier()


def evaluate_jacobian_shared(
    inputs: structured_ops.SharedPointer,
    jacobian: structured_ops.SharedPointer,
    scratch: structured_ops.SharedPointer,
    physical_warp: Int,
    physical_lane: Int,
    active_cells: Int,
):
    if (
        physical_warp < structured_ops.DagWarpsPerWave
        and physical_lane < active_cells
    ):
        structured_ops.dispatch_jac_slice_shared(
            physical_warp,
            inputs,
            jacobian,
            scratch,
            physical_lane,
            reproducer.redshift(),
        )
    barrier()
    if (
        physical_warp >= structured_ops.DagWarpsPerWave
        and physical_warp < structured_ops.DagWarps
        and physical_lane < active_cells
    ):
        structured_ops.dispatch_jac_slice_shared(
            physical_warp,
            inputs,
            jacobian,
            scratch,
            physical_lane,
            reproducer.redshift(),
        )
    barrier()


def evaluate_rhs_shared(
    inputs: structured_ops.SharedPointer,
    rhs: structured_ops.SharedPointer,
    scratch: structured_ops.SharedPointer,
    physical_warp: Int,
    physical_lane: Int,
    active_cells: Int,
):
    if (
        physical_warp < structured_ops.DagWarpsPerWave
        and physical_lane < active_cells
    ):
        structured_ops.dispatch_rhs_slice_shared(
            physical_warp,
            inputs,
            rhs,
            scratch,
            physical_lane,
            reproducer.redshift(),
        )
    barrier()
    if (
        physical_warp >= structured_ops.DagWarpsPerWave
        and physical_warp < structured_ops.DagWarps
        and physical_lane < active_cells
    ):
        structured_ops.dispatch_rhs_slice_shared(
            physical_warp,
            inputs,
            rhs,
            scratch,
            physical_lane,
            reproducer.redshift(),
        )
    barrier()


def structured_trial_kernel[
    StateLayout: TensorLayout,
    ErrorLayout: TensorLayout,
    InfoLayout: TensorLayout,
](
    states: TileTensor[DType.float64, StateLayout, MutAnyOrigin],
    candidates: TileTensor[DType.float64, StateLayout, MutAnyOrigin],
    errors: TileTensor[DType.float64, ErrorLayout, MutAnyOrigin],
    info_output: TileTensor[DType.int32, InfoLayout, MutAnyOrigin],
    num_cells: Int,
    h: Float64,
):
    comptime assert states.flat_rank == 2
    comptime assert candidates.flat_rank == 2
    comptime assert errors.flat_rank == 1
    comptime assert info_output.flat_rank == 1
    structured_ops.validate_config()
    var base = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells, reproducer.Neqs]())
    var candidate_stage = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells, reproducer.Neqs]())
    var jacobian = external_memory[
        Scalar[DType.float64],
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()
    var scratch = jacobian + structured_ops.DagScratchOffset
    var rhs = jacobian + structured_ops.RhsOffset
    var dag_inputs = jacobian + structured_ops.DagInputOffset
    var stage_y = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells, reproducer.Neqs]())
    var k1 = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.BatchCells, reproducer.Neqs]())
    var k2 = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.BatchCells, reproducer.Neqs]())
    var work = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.BatchCells, reproducer.Neqs]())
    var pivots = stack_allocation[
        DType.int32, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells, reproducer.Neqs]())
    var infos = stack_allocation[
        DType.int32, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells]())

    var tid = thread_idx.x
    var physical_warp = tid // WARP_SIZE
    var physical_lane = tid % WARP_SIZE
    var warp_id = (
        physical_warp * structured_ops.LuGroupsPerWarp
        + physical_lane // structured_ops.LuGroupWidth
    )
    var lane = physical_lane % structured_ops.LuGroupWidth
    var tile_begin = block_idx.x * structured_ops.TileCells
    var active_cells = min(structured_ops.TileCells, num_cells - tile_begin)
    if tid < structured_ops.TileCells and tid < active_cells:
        for component in range(reproducer.Neqs):
            base[tid, component] = rebind[Scalar[DType.float64]](
                states[tile_begin + tid, component]
            )
    barrier()

    comptime gamma = 0.292893218813452
    comptime a21 = 2.0000000000000036
    comptime a31 = 6.828427124746214
    comptime a32 = 3.4142135623731007
    comptime c21 = -6.828427124746214
    comptime c31 = -10.949747468305889
    comptime c32 = -7.535533905932761
    comptime b1 = 6.828427124746214
    comptime b2 = 3.414213562373101
    comptime e1 = -0.23570226039551292
    comptime e2 = -0.23570226039551567
    comptime e3 = -0.13807118745769906

    prepare_dag_inputs(base, dag_inputs, tid, active_cells)
    evaluate_jacobian_shared(
        dag_inputs, jacobian, scratch, physical_warp, physical_lane, active_cells
    )
    var tile_cell = warp_id
    var participating = tile_cell < active_cells
    comptime matrix_values = reproducer.Neqs * reproducer.Neqs
    if participating:
        for column in range(
            lane, reproducer.Neqs, structured_ops.LuGroupWidth
        ):
            for row in range(reproducer.Neqs):
                var index = (
                    tile_cell * matrix_values
                    + row * reproducer.Neqs
                    + column
                )
                jacobian[index] = -jacobian[index]
                if row == column:
                    jacobian[index] += 1.0 / (h * gamma)
    barrier()
    factorize_shared(
        jacobian, participating, tile_cell, lane, pivots, infos
    )

    prepare_dag_inputs(base, dag_inputs, tid, active_cells)
    evaluate_rhs_shared(
        dag_inputs, rhs, scratch, physical_warp, physical_lane, active_cells
    )
    solve_shared(
        jacobian, rhs, participating, tile_cell, lane, pivots, infos
    )
    if participating:
        for component in range(
            lane, reproducer.Neqs, structured_ops.LuGroupWidth
        ):
            var solution = rebind[Scalar[DType.float64]](
                rhs[tile_cell * reproducer.Neqs + component]
            )
            k1[tile_cell, component] = solution
            stage_y[tile_cell, component] = (
                rebind[Scalar[DType.float64]](base[tile_cell, component])
                + reproducer.portable_mul(a21, solution)
            )
            k2[tile_cell, component] = (c21 / h) * solution
    barrier()

    prepare_dag_inputs(stage_y, dag_inputs, tid, active_cells)
    evaluate_rhs_shared(
        dag_inputs, rhs, scratch, physical_warp, physical_lane, active_cells
    )
    if participating:
        for component in range(
            lane, reproducer.Neqs, structured_ops.LuGroupWidth
        ):
            rhs[tile_cell * reproducer.Neqs + component] = (
                rhs[tile_cell * reproducer.Neqs + component]
                + rebind[Scalar[DType.float64]](k2[tile_cell, component])
            )
    barrier()
    solve_shared(
        jacobian, rhs, participating, tile_cell, lane, pivots, infos
    )
    if participating:
        for component in range(
            lane, reproducer.Neqs, structured_ops.LuGroupWidth
        ):
            var solution = rebind[Scalar[DType.float64]](
                rhs[tile_cell * reproducer.Neqs + component]
            )
            k2[tile_cell, component] = solution
            stage_y[tile_cell, component] = (
                rebind[Scalar[DType.float64]](base[tile_cell, component])
                + reproducer.portable_mul(
                    a31,
                    rebind[Scalar[DType.float64]](k1[tile_cell, component]),
                )
                + reproducer.portable_mul(a32, solution)
            )
            work[tile_cell, component] = (
                reproducer.portable_mul(
                    c31,
                    rebind[Scalar[DType.float64]](k1[tile_cell, component]),
                )
                + reproducer.portable_mul(c32, solution)
            ) / h
    barrier()

    prepare_dag_inputs(stage_y, dag_inputs, tid, active_cells)
    evaluate_rhs_shared(
        dag_inputs, rhs, scratch, physical_warp, physical_lane, active_cells
    )
    if participating:
        for component in range(
            lane, reproducer.Neqs, structured_ops.LuGroupWidth
        ):
            rhs[tile_cell * reproducer.Neqs + component] = (
                rhs[tile_cell * reproducer.Neqs + component]
                + rebind[Scalar[DType.float64]](work[tile_cell, component])
            )
    barrier()
    solve_shared(
        jacobian, rhs, participating, tile_cell, lane, pivots, infos
    )
    if participating:
        for component in range(
            lane, reproducer.Neqs, structured_ops.LuGroupWidth
        ):
            var first = rebind[Scalar[DType.float64]](
                k1[tile_cell, component]
            )
            var second = rebind[Scalar[DType.float64]](
                k2[tile_cell, component]
            )
            var third = rebind[Scalar[DType.float64]](
                rhs[tile_cell * reproducer.Neqs + component]
            )
            candidate_stage[tile_cell, component] = (
                rebind[Scalar[DType.float64]](base[tile_cell, component])
                + reproducer.portable_mul(b1, first)
                + reproducer.portable_mul(b2, second)
                + third
            )
            work[tile_cell, component] = (
                reproducer.portable_mul(e1, first)
                + reproducer.portable_mul(e2, second)
                + reproducer.portable_mul(e3, third)
            )
    barrier()
    if participating and lane == 0:
        var sum_squared = 0.0
        for component in range(reproducer.Neqs):
            var old_value = rebind[Scalar[DType.float64]](
                base[tile_cell, component]
            )
            var new_value = rebind[Scalar[DType.float64]](
                candidate_stage[tile_cell, component]
            )
            var rtol = (
                reproducer.RtolEnergy
                if component == reproducer.NetIenuc
                else reproducer.RtolSpec
            )
            var atol = (
                reproducer.AtolEnergy
                if component == reproducer.NetIenuc
                else reproducer.AtolSpec
            )
            var scale = atol + reproducer.portable_mul(
                rtol, max(abs(old_value), abs(new_value))
            )
            var term = rebind[Scalar[DType.float64]](
                work[tile_cell, component]
            ) / scale
            sum_squared += reproducer.portable_mul(term, term)
        errors[tile_begin + tile_cell] = reproducer.portable_sqrt(
            sum_squared / Float64(reproducer.Neqs)
        )
        info_output[tile_begin + tile_cell] = rebind[Scalar[DType.int32]](
            infos[tile_cell]
        )
    barrier()

    if tid < active_cells:
        for component in range(reproducer.Neqs):
            candidates[tile_begin + tid, component] = rebind[
                Scalar[DType.float64]
            ](candidate_stage[tid, component])


def structured_chemistry_kernel[
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
    structured_ops.validate_config()
    var base = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells, reproducer.Neqs]())
    var candidate_stage = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells, reproducer.Neqs]())
    var jacobian = external_memory[
        Scalar[DType.float64],
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()
    var scratch = jacobian + structured_ops.DagScratchOffset
    var rhs = jacobian + structured_ops.RhsOffset
    var dag_inputs = jacobian + structured_ops.DagInputOffset
    var stage_y = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells, reproducer.Neqs]())
    var k1 = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.BatchCells, reproducer.Neqs]())
    var k2 = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.BatchCells, reproducer.Neqs]())
    var work = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.BatchCells, reproducer.Neqs]())
    var pivots = stack_allocation[
        DType.int32, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells, reproducer.Neqs]())
    var infos = stack_allocation[
        DType.int32, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells]())
    var participants = stack_allocation[
        DType.int32, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells]())
    var trial_errors = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells]())
    var control_i32 = stack_allocation[
        DType.int32, address_space=AddressSpace.SHARED
    ](row_major[4]())
    var control_f64 = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[3]())

    var tid = thread_idx.x
    var physical_warp = tid // WARP_SIZE
    var physical_lane = tid % WARP_SIZE
    var warp_id = (
        physical_warp * structured_ops.LuGroupsPerWarp
        + physical_lane // structured_ops.LuGroupWidth
    )
    var lane = physical_lane % structured_ops.LuGroupWidth
    var tile_begin = block_idx.x * structured_ops.TileCells
    var active_cells = min(structured_ops.TileCells, num_cells - tile_begin)
    if tid < structured_ops.TileCells:
        var participating = False
        if tid < active_cells:
            var cell = tile_begin + tid
            var collapse = reproducer_gpu.load_collapse(
                state_tensor, steps, stats, cell
            )
            if collapse.completed_steps == completed_global_steps:
                if collapse.time != grid_time:
                    failure[0] = Int32(reproducer.BadInputs)
                else:
                    var local_dt = reproducer.collapse_timestep(collapse)
                    if local_dt >= 10.0:
                        var old_density = collapse.density_driver
                        collapse.density_driver += reproducer.portable_mul(
                            dt_grid, collapse.density_driver
                        ) / (local_dt / reproducer.TffReduc)
                        if (
                            reproducer.valid_positive(
                                collapse.density_driver
                            )
                            and collapse.density_driver <= 2.0e-6
                        ):
                            var ratio = collapse.density_driver / old_density
                            for species in range(reproducer.NumSpec):
                                collapse.current.xn[species] *= ratio
                            collapse.current.rho *= ratio
                            reproducer.eos_rt(collapse.current)
                            participating = True
            for component in range(reproducer.NumSpec):
                base[tid, component] = collapse.current.xn[component]
            base[tid, reproducer.NetIenuc] = collapse.current.e
            reproducer_gpu.store_collapse(
                state_tensor, steps, stats, cell, collapse
            )
        participants[tid] = Int32(participating)
    barrier()
    if tid == 0:
        var count = 0
        for cell in range(active_cells):
            count += Int(rebind[Scalar[DType.int32]](participants[cell]))
        control_i32[0] = Int32(count)
        control_i32[2] = Int32(0)
        control_i32[3] = Int32(0)
        control_f64[1] = 0.0
        control_f64[2] = 1.0
    barrier()

    comptime uround = 1.0e-16
    comptime fac_min = 0.2
    comptime fac_max = 6.0
    comptime safe = 0.9
    comptime max_steps = 10000000
    comptime gamma = 0.292893218813452
    comptime a21 = 2.0000000000000036
    comptime a31 = 6.828427124746214
    comptime a32 = 3.4142135623731007
    comptime c21 = -6.828427124746214
    comptime c31 = -10.949747468305889
    comptime c32 = -7.535533905932761
    comptime b1 = 6.828427124746214
    comptime b2 = 3.414213562373101
    comptime e1 = -0.23570226039551292
    comptime e2 = -0.23570226039551567
    comptime e3 = -0.13807118745769906
    var h = dt_grid
    var x = 0.0
    var reject = False
    var nsing = 0
    var n_step = 0
    var n_accept = 0
    var status = reproducer.Success

    while True:
        if (
            rebind[Scalar[DType.int32]](control_i32[0]) == 0
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
        if tid < structured_ops.TileCells:
            trial_errors[tid] = 0.0
        if tid == 0:
            control_i32[1] = Int32(0)
        barrier()

        prepare_dag_inputs(base, dag_inputs, tid, active_cells)
        evaluate_jacobian_shared(
            dag_inputs,
            jacobian,
            scratch,
            physical_warp,
            physical_lane,
            active_cells,
        )
        var tile_cell = warp_id
        var participating = False
        if tile_cell < active_cells:
            participating = rebind[Scalar[DType.int32]](
                participants[tile_cell]
            ) != 0
        comptime matrix_values = reproducer.Neqs * reproducer.Neqs
        if participating:
            for column in range(
                lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                for row in range(reproducer.Neqs):
                    var index = (
                        tile_cell * matrix_values
                        + row * reproducer.Neqs
                        + column
                    )
                    jacobian[index] = -jacobian[index]
                    if row == column:
                        jacobian[index] += 1.0 / (h * gamma)
        barrier()
        factorize_shared(
            jacobian, participating, tile_cell, lane, pivots, infos
        )
        barrier()
        if tid == 0:
            for owner in range(active_cells):
                if rebind[Scalar[DType.int32]](infos[owner]) != 0:
                    control_i32[1] = Int32(1)
        barrier()

        prepare_dag_inputs(base, dag_inputs, tid, active_cells)
        evaluate_rhs_shared(
            dag_inputs,
            rhs,
            scratch,
            physical_warp,
            physical_lane,
            active_cells,
        )
        solve_shared(
            jacobian, rhs, participating, tile_cell, lane, pivots, infos
        )
        if participating:
            for component in range(
                lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                var solution = rebind[Scalar[DType.float64]](
                    rhs[tile_cell * reproducer.Neqs + component]
                )
                k1[tile_cell, component] = solution
                stage_y[tile_cell, component] = (
                    rebind[Scalar[DType.float64]](
                        base[tile_cell, component]
                    )
                    + reproducer.portable_mul(a21, solution)
                )
                k2[tile_cell, component] = (c21 / h) * solution
        barrier()

        prepare_dag_inputs(stage_y, dag_inputs, tid, active_cells)
        evaluate_rhs_shared(
            dag_inputs,
            rhs,
            scratch,
            physical_warp,
            physical_lane,
            active_cells,
        )
        if participating:
            for component in range(
                lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                rhs[tile_cell * reproducer.Neqs + component] = (
                    rhs[tile_cell * reproducer.Neqs + component]
                    + rebind[Scalar[DType.float64]](
                        k2[tile_cell, component]
                    )
                )
        barrier()
        solve_shared(
            jacobian, rhs, participating, tile_cell, lane, pivots, infos
        )
        if participating:
            for component in range(
                lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                var solution = rebind[Scalar[DType.float64]](
                    rhs[tile_cell * reproducer.Neqs + component]
                )
                k2[tile_cell, component] = solution
                stage_y[tile_cell, component] = (
                    rebind[Scalar[DType.float64]](
                        base[tile_cell, component]
                    )
                    + reproducer.portable_mul(
                        a31,
                        rebind[Scalar[DType.float64]](
                            k1[tile_cell, component]
                        ),
                    )
                    + reproducer.portable_mul(a32, solution)
                )
                work[tile_cell, component] = (
                    reproducer.portable_mul(
                        c31,
                        rebind[Scalar[DType.float64]](
                            k1[tile_cell, component]
                        ),
                    )
                    + reproducer.portable_mul(c32, solution)
                ) / h
        barrier()

        prepare_dag_inputs(stage_y, dag_inputs, tid, active_cells)
        evaluate_rhs_shared(
            dag_inputs,
            rhs,
            scratch,
            physical_warp,
            physical_lane,
            active_cells,
        )
        if participating:
            for component in range(
                lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                rhs[tile_cell * reproducer.Neqs + component] = (
                    rhs[tile_cell * reproducer.Neqs + component]
                    + rebind[Scalar[DType.float64]](
                        work[tile_cell, component]
                    )
                )
        barrier()
        solve_shared(
            jacobian, rhs, participating, tile_cell, lane, pivots, infos
        )
        if participating:
            for component in range(
                lane, reproducer.Neqs, structured_ops.LuGroupWidth
            ):
                var first = rebind[Scalar[DType.float64]](
                    k1[tile_cell, component]
                )
                var second = rebind[Scalar[DType.float64]](
                    k2[tile_cell, component]
                )
                var third = rebind[Scalar[DType.float64]](
                    rhs[tile_cell * reproducer.Neqs + component]
                )
                candidate_stage[tile_cell, component] = (
                    rebind[Scalar[DType.float64]](
                        base[tile_cell, component]
                    )
                    + reproducer.portable_mul(b1, first)
                    + reproducer.portable_mul(b2, second)
                    + third
                )
                work[tile_cell, component] = (
                    reproducer.portable_mul(e1, first)
                    + reproducer.portable_mul(e2, second)
                    + reproducer.portable_mul(e3, third)
                )
        barrier()
        if participating and lane == 0:
            var sum_squared = 0.0
            for component in range(reproducer.Neqs):
                var old_value = rebind[Scalar[DType.float64]](
                    base[tile_cell, component]
                )
                var new_value = rebind[Scalar[DType.float64]](
                    candidate_stage[tile_cell, component]
                )
                var rtol = (
                    reproducer.RtolEnergy
                    if component == reproducer.NetIenuc
                    else reproducer.RtolSpec
                )
                var atol = (
                    reproducer.AtolEnergy
                    if component == reproducer.NetIenuc
                    else reproducer.AtolSpec
                )
                var scale = atol + reproducer.portable_mul(
                    rtol, max(abs(old_value), abs(new_value))
                )
                var term = rebind[Scalar[DType.float64]](
                    work[tile_cell, component]
                ) / scale
                sum_squared += reproducer.portable_mul(term, term)
            trial_errors[tile_cell] = reproducer.portable_sqrt(
                sum_squared / Float64(reproducer.Neqs)
            )
        barrier()

        if rebind[Scalar[DType.int32]](control_i32[1]) != 0:
            if tid == 0:
                control_i32[2] = rebind[Scalar[DType.int32]](
                    control_i32[2]
                ) + Int32(1)
            nsing += 1
            if nsing >= 5:
                status = reproducer.LuDecompositionError
            else:
                h *= 0.5
                reject = True
            continue

        if tid == 0:
            var tile_error = 0.0
            for cell in range(active_cells):
                if rebind[Scalar[DType.int32]](participants[cell]) != 0:
                    var error = rebind[Scalar[DType.float64]](
                        trial_errors[cell]
                    )
                    if not isfinite(error):
                        error = Float64.MAX
                    tile_error = max(tile_error, error)
            control_f64[0] = tile_error
        barrier()
        var err_tile = rebind[Scalar[DType.float64]](control_f64[0])
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
                        (
                            rebind[Scalar[DType.float64]](control_f64[1])
                            / h
                        )
                        * gpu_cbrt_nonnegative(
                            (err_tile * err_tile)
                            / rebind[Scalar[DType.float64]](control_f64[2])
                        )
                        / safe,
                    ),
                )
                hnew = h / max(fac_step, facgus)
            if tid < active_cells and rebind[Scalar[DType.int32]](
                participants[tid]
            ) != 0:
                for component in range(reproducer.Neqs):
                    base[tid, component] = rebind[Scalar[DType.float64]](
                        candidate_stage[tid, component]
                    )
            barrier()
            if tid == 0:
                control_f64[1] = h
                control_f64[2] = max(1.0e-2, err_tile)
            x = dt_grid if final_trial else x + h
            hnew = min(abs(hnew), dt_grid)
            if reject:
                hnew = min(hnew, abs(h))
            reject = False
            h = hnew
        else:
            if tid == 0 and n_accept >= 1:
                control_i32[3] = rebind[Scalar[DType.int32]](
                    control_i32[3]
                ) + Int32(1)
            reject = True
            h = hnew

    barrier()
    if tid < active_cells:
        var cell = tile_begin + tid
        var collapse = reproducer_gpu.load_collapse(
            state_tensor, steps, stats, cell
        )
        var participating = rebind[Scalar[DType.int32]](
            participants[tid]
        ) != 0
        if status == reproducer.Success and participating:
            for species in range(reproducer.NumSpec):
                collapse.current.xn[species] = rebind[
                    Scalar[DType.float64]
                ](base[tid, species])
            collapse.current.e = rebind[Scalar[DType.float64]](
                base[tid, reproducer.NetIenuc]
            )
            reproducer.floor_and_normalize_number_densities(
                collapse.current
            )
            reproducer.balance_charge(collapse.current)
            reproducer.floor_and_normalize_number_densities(
                collapse.current
            )
            reproducer.eos_re(collapse.current)
            collapse.time = next_grid_time
            collapse.completed_steps = completed_global_steps + 1
            collapse.stats.internal_steps += n_step
            collapse.stats.rhs_calls += 3 * n_step
            collapse.stats.jacobian_calls += n_step + Int(
                rebind[Scalar[DType.int32]](control_i32[2])
            )
            collapse.stats.decompositions += n_step
            collapse.stats.linear_solves += 3 * n_step
            collapse.stats.accepted_steps += n_accept
            collapse.stats.rejected_steps += Int(
                rebind[Scalar[DType.int32]](control_i32[3])
            )
        reproducer_gpu.store_collapse(
            state_tensor, steps, stats, cell, collapse
        )
    barrier()
    if tid == 0:
        integrated_by_tile[block_idx.x] = Int32(
            Int(rebind[Scalar[DType.int32]](control_i32[0]))
            if status == reproducer.Success
            else 0
        )
        if status != reproducer.Success:
            failure[0] = Int32(status)

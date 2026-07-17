# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: Four-batch structured ROS2S trial microkernel with transactional output.

import reproducer
import structured_ops
import warp_lu
from layout import TensorLayout, TileTensor, row_major, stack_allocation
from std.gpu import WARP_SIZE, barrier, block_idx, thread_idx
from std.gpu.memory import AddressSpace
from std.math import abs, max, min


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


def evaluate_jacobian_batch[
    BaseLayout: TensorLayout,
    JacobianLayout: TensorLayout,
](
    base: TileTensor[
        DType.float64,
        BaseLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    jacobian: TileTensor[
        DType.float64,
        JacobianLayout,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ],
    batch: Int,
    warp_id: Int,
    lane: Int,
    active_cells: Int,
):
    comptime assert base.flat_rank == 2
    comptime assert jacobian.flat_rank == 3
    var tile_cell = batch * structured_ops.BatchCells + lane
    if (
        warp_id < structured_ops.NumWarps
        and lane < structured_ops.BatchCells
        and tile_cell < active_cells
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
                jacobian[lane, row, column] = reproducer.mget(
                    local, row, column
                )


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
    batch: Int,
    warp_id: Int,
    lane: Int,
    active_cells: Int,
):
    comptime assert inputs.flat_rank == 2
    comptime assert outputs.flat_rank == 2
    var tile_cell = batch * structured_ops.BatchCells + lane
    if (
        warp_id < structured_ops.NumWarps
        and lane < structured_ops.BatchCells
        and tile_cell < active_cells
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
        var second = warp_id + structured_ops.NumWarps
        if second < reproducer.NumSpec:
            outputs[lane, second] = reproducer.vget(local, second)
        if warp_id == 6:
            outputs[lane, reproducer.NetIenuc] = reproducer.vget(
                local, reproducer.NetIenuc
            )


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
    var jacobian = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](
        row_major[
            structured_ops.BatchCells, reproducer.Neqs, reproducer.Neqs
        ]()
    )
    var stage_y = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.TileCells, reproducer.Neqs]())
    var rhs = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.BatchCells, reproducer.Neqs]())
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
    ](row_major[structured_ops.NumWarps, reproducer.Neqs]())
    var exchange = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.NumWarps, 2]())
    var infos = stack_allocation[
        DType.int32, address_space=AddressSpace.SHARED
    ](row_major[structured_ops.NumWarps]())

    var tid = thread_idx.x
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
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

    for batch in range(structured_ops.BatchesPerTile):
        evaluate_jacobian_batch(
            base, jacobian, batch, warp_id, lane, active_cells
        )
        barrier()
        var owned = warp_id < structured_ops.NumWarps
        var tile_cell = batch * structured_ops.BatchCells + warp_id
        var participating = owned and tile_cell < active_cells
        var op = warp_lu.WarpLuOp()
        if participating and lane < reproducer.Neqs:
            for row in range(reproducer.Neqs):
                op.column[row] = -rebind[Scalar[DType.float64]](
                    jacobian[warp_id, row, lane]
                )
            op.column[lane] += 1.0 / (h * gamma)
        factorize_columns(
            op, participating, warp_id, lane, pivots, exchange, infos
        )

        evaluate_rhs_batch(base, rhs, batch, warp_id, lane, active_cells)
        barrier()
        if participating and lane < reproducer.Neqs:
            op.rhs = rebind[Scalar[DType.float64]](rhs[warp_id, lane])
        solve_columns(
            op, participating, warp_id, lane, pivots, exchange, infos
        )
        if participating and lane < reproducer.Neqs:
            k1[warp_id, lane] = op.rhs
            stage_y[tile_cell, lane] = (
                rebind[Scalar[DType.float64]](base[tile_cell, lane])
                + reproducer.portable_mul(a21, op.rhs)
            )
            k2[warp_id, lane] = (c21 / h) * op.rhs
        barrier()

        evaluate_rhs_batch(
            stage_y, rhs, batch, warp_id, lane, active_cells
        )
        barrier()
        if participating and lane < reproducer.Neqs:
            op.rhs = (
                rebind[Scalar[DType.float64]](k2[warp_id, lane])
                + rebind[Scalar[DType.float64]](rhs[warp_id, lane])
            )
        solve_columns(
            op, participating, warp_id, lane, pivots, exchange, infos
        )
        if participating and lane < reproducer.Neqs:
            k2[warp_id, lane] = op.rhs
            stage_y[tile_cell, lane] = (
                rebind[Scalar[DType.float64]](base[tile_cell, lane])
                + reproducer.portable_mul(
                    a31,
                    rebind[Scalar[DType.float64]](k1[warp_id, lane]),
                )
                + reproducer.portable_mul(a32, op.rhs)
            )
            work[warp_id, lane] = (
                reproducer.portable_mul(
                    c31,
                    rebind[Scalar[DType.float64]](k1[warp_id, lane]),
                )
                + reproducer.portable_mul(c32, op.rhs)
            ) / h
        barrier()

        evaluate_rhs_batch(
            stage_y, rhs, batch, warp_id, lane, active_cells
        )
        barrier()
        if participating and lane < reproducer.Neqs:
            op.rhs = (
                rebind[Scalar[DType.float64]](work[warp_id, lane])
                + rebind[Scalar[DType.float64]](rhs[warp_id, lane])
            )
        solve_columns(
            op, participating, warp_id, lane, pivots, exchange, infos
        )
        if participating and lane < reproducer.Neqs:
            var first = rebind[Scalar[DType.float64]](k1[warp_id, lane])
            var second = rebind[Scalar[DType.float64]](k2[warp_id, lane])
            candidate_stage[tile_cell, lane] = (
                rebind[Scalar[DType.float64]](base[tile_cell, lane])
                + reproducer.portable_mul(b1, first)
                + reproducer.portable_mul(b2, second)
                + op.rhs
            )
            work[warp_id, lane] = (
                reproducer.portable_mul(e1, first)
                + reproducer.portable_mul(e2, second)
                + reproducer.portable_mul(e3, op.rhs)
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
                    work[warp_id, component]
                ) / scale
                sum_squared += reproducer.portable_mul(term, term)
            errors[tile_begin + tile_cell] = reproducer.portable_sqrt(
                sum_squared / Float64(reproducer.Neqs)
            )
            info_output[tile_begin + tile_cell] = infos[warp_id]
        barrier()

    if tid < active_cells:
        for component in range(reproducer.Neqs):
            candidates[tile_begin + tid, component] = rebind[
                Scalar[DType.float64]
            ](candidate_stage[tid, component])

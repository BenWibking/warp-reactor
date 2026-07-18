# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: Column-owned warp-cooperative 15x15 LU factorization and solve.

from layout import TensorLayout, TileTensor, row_major, stack_allocation
from std.gpu import WARP_SIZE, barrier, thread_idx
from std.gpu.memory import AddressSpace
from std.math import abs


comptime Neqs = 15
comptime OwnerWarps = 8
comptime MaxSharedBytes = 48 * 1024
comptime PivotBytes = OwnerWarps * Neqs * 4
comptime ExchangeBytes = OwnerWarps * 2 * 8
comptime InfoBytes = OwnerWarps * 4
comptime LuSharedBytes = PivotBytes + ExchangeBytes + InfoBytes


struct WarpLuOp:
    var column: InlineArray[Float64, Neqs]
    var rhs: Float64
    var pivot_row: Int

    def __init__(out self):
        self.column = InlineArray[Float64, Neqs](fill=0.0)
        self.rhs = 0.0
        self.pivot_row = 0

    def select_pivot(mut self, k: Int):
        self.pivot_row = k
        var maximum = abs(self.column[k])
        for row in range(k + 1, Neqs):
            var value = abs(self.column[row])
            if value > maximum:
                maximum = value
                self.pivot_row = row

    def swap_rows(mut self, k: Int, pivot: Int):
        if pivot != k:
            var temporary = self.column[pivot]
            self.column[pivot] = self.column[k]
            self.column[k] = temporary


def warp_lu_solve_kernel[
    MatrixLayout: TensorLayout,
    VectorLayout: TensorLayout,
    InfoLayout: TensorLayout,
](
    matrices: TileTensor[DType.float64, MatrixLayout, MutAnyOrigin],
    vectors: TileTensor[DType.float64, VectorLayout, MutAnyOrigin],
    lu_output: TileTensor[DType.float64, MatrixLayout, MutAnyOrigin],
    solution_output: TileTensor[DType.float64, VectorLayout, MutAnyOrigin],
    info_output: TileTensor[DType.int32, InfoLayout, MutAnyOrigin],
    num_cases: Int,
):
    comptime assert LuSharedBytes <= MaxSharedBytes
    comptime assert matrices.flat_rank == 2
    comptime assert vectors.flat_rank == 2
    comptime assert lu_output.flat_rank == 2
    comptime assert solution_output.flat_rank == 2
    comptime assert info_output.flat_rank == 1
    var pivots = stack_allocation[
        DType.int32, address_space=AddressSpace.SHARED
    ](row_major[OwnerWarps, Neqs]())
    var exchange = stack_allocation[
        DType.float64, address_space=AddressSpace.SHARED
    ](row_major[OwnerWarps, 2]())
    var infos = stack_allocation[
        DType.int32, address_space=AddressSpace.SHARED
    ](row_major[OwnerWarps]())

    var physical_warp = thread_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE
    var owned = physical_warp < OwnerWarps and physical_warp < num_cases
    var active_column = owned and lane < Neqs
    var op = WarpLuOp()
    if active_column:
        for row in range(Neqs):
            op.column[row] = rebind[Scalar[DType.float64]](
                matrices[physical_warp, row * Neqs + lane]
            )
        op.rhs = rebind[Scalar[DType.float64]](
            vectors[physical_warp, lane]
        )
    if physical_warp < OwnerWarps and lane == 0:
        infos[physical_warp] = Int32(0)
    barrier()

    for k in range(Neqs - 1):
        if owned and lane == k:
            op.select_pivot(k)
            pivots[physical_warp, k] = Int32(op.pivot_row)
        barrier()
        var pivot = k
        if owned:
            pivot = Int(
                rebind[Scalar[DType.int32]](pivots[physical_warp, k])
            )
        if active_column and lane >= k:
            op.swap_rows(k, pivot)
        barrier()
        if owned and lane == k:
            if op.column[k] == 0.0:
                infos[physical_warp] = Int32(k + 1)
            else:
                var scale = -1.0 / op.column[k]
                for row in range(k + 1, Neqs):
                    op.column[row] *= scale
        barrier()
        for row in range(k + 1, Neqs):
            if owned and lane == k:
                exchange[physical_warp, 0] = (
                    op.column[row] if op.column[k] != 0.0 else 0.0
                )
            barrier()
            if owned:
                var multiplier = rebind[Scalar[DType.float64]](
                    exchange[physical_warp, 0]
                )
                if active_column and lane > k:
                    op.column[row] += op.column[k] * multiplier
            barrier()

    if owned and lane == Neqs - 1 and op.column[Neqs - 1] == 0.0:
        infos[physical_warp] = Int32(Neqs)
    barrier()

    # LINPACK forward substitution with the distributed RHS vector.
    for k in range(Neqs - 1):
        var pivot = k
        if owned:
            pivot = Int(
                rebind[Scalar[DType.int32]](pivots[physical_warp, k])
            )
        if owned and lane == k:
            exchange[physical_warp, 0] = op.rhs
        if owned and lane == pivot:
            exchange[physical_warp, 1] = op.rhs
        barrier()
        if active_column:
            var x_k = rebind[Scalar[DType.float64]](
                exchange[physical_warp, 0]
            )
            var x_pivot = rebind[Scalar[DType.float64]](
                exchange[physical_warp, 1]
            )
            if lane == pivot:
                op.rhs = x_k
            if lane == k:
                op.rhs = x_pivot
        barrier()
        if owned and lane == k:
            exchange[physical_warp, 0] = op.rhs
        barrier()
        var t = 0.0
        if owned:
            t = rebind[Scalar[DType.float64]](
                exchange[physical_warp, 0]
            )
        for row in range(k + 1, Neqs):
            if owned and lane == k:
                exchange[physical_warp, 1] = op.column[row]
            barrier()
            if owned and lane == row:
                op.rhs += t * rebind[Scalar[DType.float64]](
                    exchange[physical_warp, 1]
                )
            barrier()

    # Back substitution reuses the retained distributed LU columns.
    for reverse_k in range(Neqs):
        var k = Neqs - 1 - reverse_k
        if owned and lane == k:
            op.rhs /= op.column[k]
            exchange[physical_warp, 0] = op.rhs
        barrier()
        var negative_x = 0.0
        if owned:
            negative_x = -rebind[Scalar[DType.float64]](
                exchange[physical_warp, 0]
            )
        for row in range(k):
            if owned and lane == k:
                exchange[physical_warp, 1] = op.column[row]
            barrier()
            if owned and lane == row:
                op.rhs += negative_x * rebind[Scalar[DType.float64]](
                    exchange[physical_warp, 1]
                )
            barrier()

    if active_column:
        for row in range(Neqs):
            lu_output[physical_warp, row * Neqs + lane] = op.column[row]
        solution_output[physical_warp, lane] = op.rhs
    if owned and lane == 0:
        info_output[physical_warp] = infos[physical_warp]

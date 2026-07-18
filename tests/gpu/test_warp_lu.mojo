# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: GPU validation driver for the column-owned 15x15 LU implementation.

import reproducer
import warp_lu
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu.host import DeviceContext
from std.math import abs, max


comptime Cases = 8
comptime MatrixValues = warp_lu.Neqs * warp_lu.Neqs
comptime TestVecLayout = Layout.row_major(warp_lu.Neqs)
comptime TestMatLayout = Layout.row_major(warp_lu.Neqs, warp_lu.Neqs)
comptime TestVec = LayoutTensor[DType.float64, TestVecLayout, MutAnyOrigin]
comptime TestMat = LayoutTensor[DType.float64, TestMatLayout, MutAnyOrigin]


def initialize_case(case_id: Int, mut matrix: TestMat, mut rhs: TestVec):
    for row in range(warp_lu.Neqs):
        for column in range(warp_lu.Neqs):
            var value: Float64
            if case_id <= 3:
                value = 1.0 if row == column else 0.0
            else:
                value = (
                    Float64(
                        (row * 17 + column * 13 + case_id * 7) % 19 - 9
                    )
                    * 0.01
                )
                if row == column:
                    value += 2.0
            reproducer.mset(matrix, row, column, value)
        reproducer.vset(rhs, row, Float64((row * 5 + case_id * 3) % 11 - 5))
    if case_id == 1:
        for column in range(warp_lu.Neqs):
            reproducer.mset(matrix, warp_lu.Neqs - 1, column, 0.0)
    elif case_id == 2:
        reproducer.mset(matrix, 0, 0, 0.0)
        reproducer.mset(matrix, 1, 0, 2.0)
    elif case_id == 3:
        reproducer.mset(matrix, 0, 0, 1.0)
        reproducer.mset(matrix, 1, 0, -3.0)
        reproducer.mset(matrix, 2, 0, 3.0)
    elif case_id == 7:
        reproducer.mset(matrix, warp_lu.Neqs - 1, warp_lu.Neqs - 1, 1.0e-10)


def main() raises:
    var original = List[Float64]()
    var original_rhs = List[Float64]()
    var expected_solution = List[Float64]()
    var expected_info = List[Int]()
    for case_id in range(Cases):
        var matrix = TestMat.stack_allocation()
        var rhs = TestVec.stack_allocation()
        initialize_case(case_id, matrix, rhs)
        for row in range(warp_lu.Neqs):
            for column in range(warp_lu.Neqs):
                original.append(reproducer.mget(matrix, row, column))
            original_rhs.append(reproducer.vget(rhs, row))
        var pivots = InlineArray[Int, warp_lu.Neqs](fill=0)
        var info = reproducer.lu_decomposition(matrix, pivots)
        expected_info.append(info)
        if info == 0:
            reproducer.lu_solve(matrix, pivots, rhs)
        for row in range(warp_lu.Neqs):
            expected_solution.append(reproducer.vget(rhs, row))

    var ctx = DeviceContext()
    var matrix_buffer = ctx.enqueue_create_buffer[DType.float64](
        Cases * MatrixValues
    )
    var vector_buffer = ctx.enqueue_create_buffer[DType.float64](
        Cases * warp_lu.Neqs
    )
    var lu_buffer = ctx.enqueue_create_buffer[DType.float64](
        Cases * MatrixValues
    )
    var solution_buffer = ctx.enqueue_create_buffer[DType.float64](
        Cases * warp_lu.Neqs
    )
    var info_buffer = ctx.enqueue_create_buffer[DType.int32](Cases)
    var matrix_layout = row_major(Cases, MatrixValues)
    var vector_layout = row_major(Cases, warp_lu.Neqs)
    var info_layout = row_major(Cases)
    with matrix_buffer.map_to_host() as mapped_matrix:
        for index in range(len(original)):
            mapped_matrix[index] = original[index]
    with vector_buffer.map_to_host() as mapped_vector:
        for index in range(len(original_rhs)):
            mapped_vector[index] = original_rhs[index]

    var matrices = TileTensor(matrix_buffer, matrix_layout)
    var vectors = TileTensor(vector_buffer, vector_layout)
    var lu_output = TileTensor(lu_buffer, matrix_layout)
    var solution_output = TileTensor(solution_buffer, vector_layout)
    var info_output = TileTensor(info_buffer, info_layout)
    comptime kernel = warp_lu.warp_lu_solve_kernel[
        type_of(matrix_layout), type_of(vector_layout), type_of(info_layout)
    ]
    ctx.enqueue_function[kernel](
        matrices,
        vectors,
        lu_output,
        solution_output,
        info_output,
        Cases,
        grid_dim=1,
        block_dim=256,
    )
    ctx.synchronize()

    with info_buffer.map_to_host() as mapped_info:
        for case_id in range(Cases):
            var actual_info = Int(mapped_info[case_id])
            if actual_info != expected_info[case_id]:
                raise Error(
                    t"case_id {case_id}: LU info {actual_info}, expected {expected_info[case_id]}"
                )
    with solution_buffer.map_to_host() as mapped_solution:
        for case_id in range(Cases):
            if expected_info[case_id] != 0:
                continue
            var norm_a = 0.0
            var norm_x = 0.0
            var norm_b = 0.0
            var residual = 0.0
            for row in range(warp_lu.Neqs):
                var ax = 0.0
                var row_norm = 0.0
                for column in range(warp_lu.Neqs):
                    var a = original[case_id * MatrixValues + row * warp_lu.Neqs + column]
                    var x = mapped_solution[case_id * warp_lu.Neqs + column]
                    ax += a * x
                    row_norm += abs(a)
                    norm_x = max(norm_x, abs(x))
                var b = original_rhs[case_id * warp_lu.Neqs + row]
                residual = max(residual, abs(ax - b))
                norm_a = max(norm_a, row_norm)
                norm_b = max(norm_b, abs(b))
                var expected = expected_solution[case_id * warp_lu.Neqs + row]
                var actual = mapped_solution[case_id * warp_lu.Neqs + row]
                if abs(actual - expected) > 1.0e-10 * max(1.0, abs(expected)):
                    raise Error(
                        t"case_id {case_id}: solution mismatch at row {row}: actual={actual}, expected={expected}"
                    )
            var scale = max(1.0, norm_a * norm_x + norm_b)
            if residual > 1.0e-11 * scale:
                raise Error(t"case_id {case_id}: residual bound failed")
    print("PASS: column-owned warp LU cases and residual bounds")

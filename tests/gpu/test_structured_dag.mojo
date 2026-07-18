# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: GPU numerical driver for the eight-path structured DAG dispatch.

import reproducer
import structured_ops
from layout import TileTensor, row_major
from std.gpu.host import DeviceContext
from std.math import abs, max


comptime Cells = structured_ops.BatchCells


def main() raises:
    var state_values = List[Float64]()
    for cell in range(Cells):
        var state = reproducer.make_initial_state()
        var scale = 1.0 + Float64(cell) * 0.015625
        for species in range(reproducer.NumSpec):
            state_values.append(state.xn[species] * scale)
        state_values.append(state.e * scale)

    var ctx = DeviceContext()
    var states_buffer = ctx.enqueue_create_buffer[DType.float64](
        Cells * reproducer.Neqs
    )
    var jac_buffer = ctx.enqueue_create_buffer[DType.float64](
        Cells * reproducer.Neqs * reproducer.Neqs
    )
    var rhs_buffer = ctx.enqueue_create_buffer[DType.float64](
        Cells * reproducer.Neqs
    )
    var states_layout = row_major(Cells, reproducer.Neqs)
    var jac_layout = row_major(
        Cells, reproducer.Neqs * reproducer.Neqs
    )
    var rhs_layout = row_major(Cells, reproducer.Neqs)
    with states_buffer.map_to_host() as mapped_states:
        for index in range(len(state_values)):
            mapped_states[index] = state_values[index]
    var states = TileTensor(states_buffer, states_layout)
    var jacobians = TileTensor(jac_buffer, jac_layout)
    var right_hand_sides = TileTensor(rhs_buffer, rhs_layout)
    comptime kernel = structured_ops.dag_slice_kernel[
        type_of(states_layout), type_of(jac_layout), type_of(rhs_layout)
    ]
    ctx.enqueue_function[kernel](
        states,
        jacobians,
        right_hand_sides,
        Cells,
        grid_dim=1,
        block_dim=256,
    )
    ctx.synchronize()

    with jac_buffer.map_to_host() as mapped_jac:
        with rhs_buffer.map_to_host() as mapped_rhs:
            for cell in range(Cells):
                var y = reproducer.VecTensor.stack_allocation()
                for component in range(reproducer.Neqs):
                    reproducer.vset(
                        y,
                        component,
                        state_values[cell * reproducer.Neqs + component],
                    )
                var state = reproducer.burn_state_from_y(y)
                var expected_jac = reproducer.MatTensor.stack_allocation()
                var expected_rhs = reproducer.VecTensor.stack_allocation()
                reproducer.zero_mat(expected_jac)
                reproducer.zero_vec(expected_rhs)
                reproducer.actual_jac(state, expected_jac)
                reproducer.actual_rhs(state, expected_rhs)
                for row in range(reproducer.Neqs):
                    for column in range(reproducer.Neqs):
                        var index = (
                            cell * reproducer.Neqs * reproducer.Neqs
                            + row * reproducer.Neqs
                            + column
                        )
                        var actual = mapped_jac[index]
                        var expected = reproducer.mget(
                            expected_jac, row, column
                        )
                        if abs(actual - expected) > 2.0e-9 * max(
                            1.0e-300, abs(expected)
                        ):
                            raise Error(
                                t"cell {cell}: J mismatch at ({row}, {column})"
                            )
                for component in range(reproducer.Neqs):
                    var actual = mapped_rhs[
                        cell * reproducer.Neqs + component
                    ]
                    var expected = reproducer.vget(expected_rhs, component)
                    if abs(actual - expected) > 2.0e-9 * max(
                        1.0e-300, abs(expected)
                    ):
                        raise Error(
                            t"cell {cell}: RHS mismatch at {component}"
                        )
    print("PASS: eight-path GPU DAG dispatch")

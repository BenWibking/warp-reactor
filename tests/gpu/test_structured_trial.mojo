# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: GPU validation driver for the Hopper single-batch ROS2S trial.

import reproducer
import structured_ops
import structured_trial
from layout import TileTensor, row_major
from std.gpu.host import DeviceContext
from std.math import abs, max


comptime Cells = structured_ops.TileCells


def main() raises:
    var input_values = List[Float64]()
    var expected_candidates = List[Float64]()
    var expected_errors = List[Float64]()
    var expected_info = List[Int]()
    var h = 1.0e8
    for cell in range(Cells):
        var state = reproducer.make_initial_state()
        var scale = 1.0 + Float64(cell) * 0.0009765625
        for species in range(reproducer.NumSpec):
            state.xn[species] *= scale
            input_values.append(state.xn[species])
        state.rho = reproducer.density(state.xn)
        reproducer.eos_rt(state)
        input_values.append(state.e)
        var candidate = reproducer.BurnState()
        var stats = reproducer.IntegratorStats()
        var result = reproducer.ros2s_trial(
            state, h, 0.0, candidate, stats
        )
        expected_info.append(
            0
            if result.status == reproducer.Success
            else reproducer.Neqs
        )
        expected_errors.append(result.error)
        for species in range(reproducer.NumSpec):
            expected_candidates.append(candidate.xn[species])
        expected_candidates.append(candidate.e)

    var ctx = DeviceContext()
    var state_buffer = ctx.enqueue_create_buffer[DType.float64](
        Cells * reproducer.Neqs
    )
    var candidate_buffer = ctx.enqueue_create_buffer[DType.float64](
        Cells * reproducer.Neqs
    )
    var error_buffer = ctx.enqueue_create_buffer[DType.float64](Cells)
    var info_buffer = ctx.enqueue_create_buffer[DType.int32](Cells)
    var state_layout = row_major(Cells, reproducer.Neqs)
    var scalar_layout = row_major(Cells)
    with state_buffer.map_to_host() as mapped_states:
        for index in range(len(input_values)):
            mapped_states[index] = input_values[index]
    var states = TileTensor(state_buffer, state_layout)
    var candidates = TileTensor(candidate_buffer, state_layout)
    var errors = TileTensor(error_buffer, scalar_layout)
    var infos = TileTensor(info_buffer, scalar_layout)
    comptime kernel = structured_trial.structured_trial_kernel[
        type_of(state_layout), type_of(scalar_layout), type_of(scalar_layout)
    ]
    ctx.enqueue_function[kernel](
        states,
        candidates,
        errors,
        infos,
        Cells,
        h,
        grid_dim=1,
        block_dim=structured_ops.BlockThreads,
        shared_mem_bytes=structured_ops.DynamicSharedBytes,
    )
    ctx.synchronize()

    with candidate_buffer.map_to_host() as mapped_candidates:
        with error_buffer.map_to_host() as mapped_errors:
            with info_buffer.map_to_host() as mapped_infos:
                for cell in range(Cells):
                    if Int(mapped_infos[cell]) != expected_info[cell]:
                        raise Error(t"cell {cell}: LU status mismatch")
                    if expected_info[cell] != 0:
                        continue
                    var expected_error = expected_errors[cell]
                    var actual_error = mapped_errors[cell]
                    if abs(actual_error - expected_error) > (
                        max(
                            1.0e-10,
                            reproducer.ComparisonThermodynamicRtol
                            * abs(expected_error),
                        )
                    ):
                        raise Error(
                            t"cell {cell}: error norm mismatch: actual={actual_error}, expected={expected_error}"
                        )
                    for component in range(reproducer.Neqs):
                        var index = cell * reproducer.Neqs + component
                        var expected = expected_candidates[index]
                        var actual = mapped_candidates[index]
                        var rtol = reproducer.ComparisonThermodynamicRtol
                        var atol = reproducer.AtolEnergy
                        if component < reproducer.NumSpec:
                            rtol = reproducer.comparison_species_rtol(component)
                            atol = reproducer.AtolSpec
                        if not reproducer.nearly_equal(
                            actual, expected, rtol, atol
                        ):
                            raise Error(
                                t"cell {cell}: candidate mismatch at {component}: actual={actual}, expected={expected}"
                            )
    print("PASS: Hopper 32-cell transactional structured trial")

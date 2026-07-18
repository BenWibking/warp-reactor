# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: Hopper 32-cell structured Mojo GPU chemistry driver.

import reproducer
import reproducer_gpu
import structured_ops
import structured_trial
from layout import TileTensor, row_major
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext
from std.math import ceildiv, isfinite
from std.time import perf_counter_ns


def main() raises:
    var options = reproducer.parse_args()
    if options.show_help:
        reproducer.print_usage("reproducer_structured_gpu")
        return
    if options.driver_policy != reproducer.GridwideCta32Policy:
        raise Error(
            "reproducer_structured_gpu requires --driver-policy gridwide-cta32"
        )
    var num_cells = reproducer.checked_cell_count(options.grid_dim)
    var num_tiles = ceildiv(num_cells, structured_ops.TileCells)
    var cells = List[reproducer.CollapseState]()
    for _ in range(num_cells):
        cells.append(reproducer.make_collapse_state())

    var ctx = DeviceContext()
    var state_buffer = ctx.enqueue_create_buffer[DType.float64](
        reproducer_gpu.StateFields * num_cells
    )
    var step_buffer = ctx.enqueue_create_buffer[DType.int32](num_cells)
    var stats_buffer = ctx.enqueue_create_buffer[DType.uint64](
        reproducer_gpu.StatsFields * num_cells
    )
    var min_buffer = ctx.enqueue_create_buffer[DType.float64](num_tiles)
    var id_buffer = ctx.enqueue_create_buffer[DType.int32](num_tiles)
    var integrated_buffer = ctx.enqueue_create_buffer[DType.int32](num_tiles)
    var failure_buffer = ctx.enqueue_create_buffer[DType.int32](1)
    var state_layout = row_major(reproducer_gpu.StateFields, num_cells)
    var step_layout = row_major(num_cells)
    var stats_layout = row_major(reproducer_gpu.StatsFields, num_cells)
    var tile_layout = row_major(num_tiles)
    var failure_layout = row_major(1)
    var state_tensor = TileTensor(state_buffer, state_layout)
    var step_tensor = TileTensor(step_buffer, step_layout)
    var stats_tensor = TileTensor(stats_buffer, stats_layout)
    var minima = TileTensor(min_buffer, tile_layout)
    var minimum_cells = TileTensor(id_buffer, tile_layout)
    var integrated = TileTensor(integrated_buffer, tile_layout)
    var failure = TileTensor(failure_buffer, failure_layout)

    with state_buffer.map_to_host() as mapped_state:
        with step_buffer.map_to_host() as mapped_steps:
            with stats_buffer.map_to_host() as mapped_stats:
                for cell in range(num_cells):
                    mapped_state[
                        reproducer_gpu.RhoField * num_cells + cell
                    ] = cells[cell].current.rho
                    mapped_state[
                        reproducer_gpu.TemperatureField * num_cells + cell
                    ] = cells[cell].current.T
                    mapped_state[
                        reproducer_gpu.EnergyField * num_cells + cell
                    ] = cells[cell].current.e
                    for n in range(reproducer.NumSpec):
                        mapped_state[
                            (reproducer_gpu.SpeciesField + n) * num_cells + cell
                        ] = cells[cell].current.xn[n]
                    mapped_state[
                        reproducer_gpu.TimeField * num_cells + cell
                    ] = cells[cell].time
                    mapped_state[
                        reproducer_gpu.DriverField * num_cells + cell
                    ] = cells[cell].density_driver
                    mapped_steps[cell] = Int32(cells[cell].completed_steps)
                    for field in range(reproducer_gpu.StatsFields):
                        mapped_stats[field * num_cells + cell] = UInt64(0)

    comptime prepare = reproducer_gpu.prepare_grid_timestep_kernel[
        type_of(state_layout),
        type_of(step_layout),
        type_of(stats_layout),
        type_of(tile_layout),
        type_of(tile_layout),
        type_of(failure_layout),
    ]
    comptime chemistry = structured_trial.structured_chemistry_kernel[
        type_of(state_layout),
        type_of(step_layout),
        type_of(stats_layout),
        type_of(tile_layout),
        type_of(failure_layout),
    ]
    var completed_global_steps = 0
    var grid_time = 0.0
    var failed = reproducer.Success
    var limiting_cell = -1
    var start = perf_counter_ns()
    for step in range(reproducer.MaxCollapseSteps):
        failure_buffer.enqueue_fill(Int32(reproducer.Success))
        ctx.enqueue_function[prepare](
            state_tensor,
            step_tensor,
            stats_tensor,
            minima,
            minimum_cells,
            failure,
            num_cells,
            completed_global_steps,
            grid_time,
            step,
            Int32(options.perturb),
            grid_dim=num_tiles,
            block_dim=WARP_SIZE,
        )
        ctx.synchronize()
        var dt_grid = Float64.MAX
        limiting_cell = Int.MAX
        with min_buffer.map_to_host() as mapped_minima:
            var host_minima = TileTensor(mapped_minima, tile_layout)
            with id_buffer.map_to_host() as mapped_ids:
                var host_ids = TileTensor(mapped_ids, tile_layout)
                for tile in range(num_tiles):
                    var dt = rebind[Scalar[DType.float64]](host_minima[tile])
                    var cell = Int(
                        rebind[Scalar[DType.int32]](host_ids[tile])
                    )
                    if dt < dt_grid or (
                        dt == dt_grid and cell < limiting_cell
                    ):
                        dt_grid = dt
                        limiting_cell = cell
        with failure_buffer.map_to_host() as mapped_failure:
            failed = Int(mapped_failure[0])
        if failed != reproducer.Success or limiting_cell == Int.MAX:
            break
        var next_grid_time = grid_time + dt_grid
        if not reproducer.valid_positive(dt_grid) or not isfinite(next_grid_time):
            failed = reproducer.BadInputs
            break
        integrated_buffer.enqueue_fill(Int32(0))
        ctx.enqueue_function[chemistry](
            state_tensor,
            step_tensor,
            stats_tensor,
            integrated,
            failure,
            num_cells,
            completed_global_steps,
            grid_time,
            next_grid_time,
            dt_grid,
            grid_dim=num_tiles,
            block_dim=structured_ops.BlockThreads,
            shared_mem_bytes=structured_ops.DynamicSharedBytes,
        )
        ctx.synchronize()
        var integrated_count = 0
        with integrated_buffer.map_to_host() as mapped_integrated:
            for tile in range(num_tiles):
                integrated_count += Int(mapped_integrated[tile])
        with failure_buffer.map_to_host() as mapped_failure:
            failed = Int(mapped_failure[0])
        if failed != reproducer.Success or integrated_count == 0:
            break
        grid_time = next_grid_time
        completed_global_steps += 1
    var elapsed = Float64(perf_counter_ns() - start) * 1.0e-9
    if failed != reproducer.Success:
        raise Error(t"GPU integration failed with code {failed}")

    with state_buffer.map_to_host() as mapped_state:
        with step_buffer.map_to_host() as mapped_steps:
            with stats_buffer.map_to_host() as mapped_stats:
                for cell in range(num_cells):
                    cells[cell].current.rho = mapped_state[
                        reproducer_gpu.RhoField * num_cells + cell
                    ]
                    cells[cell].current.T = mapped_state[
                        reproducer_gpu.TemperatureField * num_cells + cell
                    ]
                    cells[cell].current.e = mapped_state[
                        reproducer_gpu.EnergyField * num_cells + cell
                    ]
                    for n in range(reproducer.NumSpec):
                        cells[cell].current.xn[n] = mapped_state[
                            (reproducer_gpu.SpeciesField + n) * num_cells + cell
                        ]
                    cells[cell].time = mapped_state[
                        reproducer_gpu.TimeField * num_cells + cell
                    ]
                    cells[cell].density_driver = mapped_state[
                        reproducer_gpu.DriverField * num_cells + cell
                    ]
                    cells[cell].completed_steps = Int(mapped_steps[cell])
                    cells[cell].stats.internal_steps = Int(mapped_stats[cell])
                    cells[cell].stats.rhs_calls = Int(
                        mapped_stats[num_cells + cell]
                    )
                    cells[cell].stats.jacobian_calls = Int(
                        mapped_stats[2 * num_cells + cell]
                    )
                    cells[cell].stats.decompositions = Int(
                        mapped_stats[3 * num_cells + cell]
                    )
                    cells[cell].stats.linear_solves = Int(
                        mapped_stats[4 * num_cells + cell]
                    )
                    cells[cell].stats.accepted_steps = Int(
                        mapped_stats[5 * num_cells + cell]
                    )
                    cells[cell].stats.rejected_steps = Int(
                        mapped_stats[6 * num_cells + cell]
                    )
    print("Primordial chemistry collapse grid with ROS2S")
    print("backend: mojo-gpu-structured-hopper")
    print("grid:", options.grid_dim, "^3 (", num_cells, "cells )")
    print("perturbations:", "enabled" if options.perturb else "disabled")
    print("driver policy: gridwide-cta32")
    print("completed global collapse steps:", completed_global_steps)
    print("limiting cell:", limiting_cell)
    if num_cells == 1:
        print("representative cell completed steps:", cells[0].completed_steps)
        print("representative cell physical time:", cells[0].time)
        print("representative cell density driver:", cells[0].density_driver)
    else:
        reproducer.print_summary(
            "cell completed steps", reproducer.summarize_cells(cells, 0)
        )
        reproducer.print_summary(
            "cell physical time", reproducer.summarize_cells(cells, 1)
        )
        reproducer.print_summary(
            "cell density driver", reproducer.summarize_cells(cells, 2)
        )
    print("wall time:", elapsed, "s")
    if num_cells == 1:
        var total_stats = reproducer.IntegratorStats()
        reproducer.add_stats(total_stats, cells[0].stats)
        print("ROS2S internal steps:", total_stats.internal_steps)
        print("ROS2S rhs calls:", total_stats.rhs_calls)
        print("ROS2S jacobian calls:", total_stats.jacobian_calls)
        print("ROS2S decompositions:", total_stats.decompositions)
        print("ROS2S linear solves:", total_stats.linear_solves)
        print(
            "ROS2S accepted/rejected:",
            total_stats.accepted_steps,
            "/",
            total_stats.rejected_steps,
        )
    else:
        reproducer.print_summary(
            "ROS2S internal steps", reproducer.summarize_cells(cells, 3)
        )
        reproducer.print_summary(
            "ROS2S rhs calls", reproducer.summarize_cells(cells, 4)
        )
        reproducer.print_summary(
            "ROS2S jacobian calls", reproducer.summarize_cells(cells, 5)
        )
        reproducer.print_summary(
            "ROS2S decompositions", reproducer.summarize_cells(cells, 6)
        )
        reproducer.print_summary(
            "ROS2S linear solves", reproducer.summarize_cells(cells, 7)
        )
        reproducer.print_summary(
            "ROS2S accepted steps", reproducer.summarize_cells(cells, 8)
        )
        reproducer.print_summary(
            "ROS2S rejected steps", reproducer.summarize_cells(cells, 9)
        )
    if options.compare_final_state_path != "":
        if not reproducer.compare_final_states_from_file(
            cells,
            options.grid_dim,
            options.compare_final_state_path,
            reproducer.GridwideCta32Policy,
        ):
            raise Error("final-state comparison failed")
    if options.write_final_state_path != "":
        reproducer.write_final_states(
            cells,
            options.grid_dim,
            options.write_final_state_path,
            reproducer.GridwideCta32Policy,
        )
    if options.grid_dim == 1:
        reproducer.print_state(cells[0].current)

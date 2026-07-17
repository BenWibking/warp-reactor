# SPDX-License-Identifier: BSD-3-Clause

import reproducer
from std.testing import TestSuite, assert_equal, assert_true


def make_cells(count: Int) -> List[reproducer.CollapseState]:
    var cells = List[reproducer.CollapseState]()
    for _ in range(count):
        cells.append(reproducer.make_collapse_state())
    return cells^


def make_mask(count: Int) -> List[Bool]:
    var mask = List[Bool]()
    for _ in range(count):
        mask.append(False)
    return mask^


def test_stable_minimum_uses_lowest_cell() raises:
    var cells = make_cells(3)
    var active = make_mask(3)
    var proposal = reproducer.prepare_grid_timestep(
        cells, active, 0, 0.0, 0, False
    )
    assert_equal(proposal.status, reproducer.Success)
    assert_equal(proposal.active_count, 3)
    assert_equal(proposal.cell, 0)
    assert_true(active[0] and active[1] and active[2])


def test_terminal_small_timestep_updates_once() raises:
    var cells = make_cells(1)
    var active = make_mask(1)
    for n in range(reproducer.NumSpec):
        cells[0].current.xn[n] = 0.0
    cells[0].current.xn[2] = 1.0e30
    cells[0].density_driver = 1.0e-10
    var proposal = reproducer.prepare_grid_timestep(
        cells, active, 0, 0.0, 0, False
    )
    assert_equal(proposal.status, reproducer.Success)
    assert_equal(proposal.active_count, 0)
    assert_true(not active[0])
    assert_true(abs(cells[0].density_driver - 1.1e-10) < 1.0e-25)


def test_invalid_density_fails() raises:
    var cells = make_cells(1)
    var active = make_mask(1)
    for n in range(reproducer.NumSpec):
        cells[0].current.xn[n] = 0.0
    var proposal = reproducer.prepare_grid_timestep(
        cells, active, 0, 0.0, 0, False
    )
    assert_equal(proposal.status, reproducer.BadInputs)
    assert_equal(proposal.active_count, 0)


def test_density_limit_stops_before_burn() raises:
    var cells = make_cells(1)
    var active = make_mask(1)
    active[0] = True
    cells[0].density_driver = 1.99e-6
    var dt = reproducer.collapse_timestep(cells[0])
    var participants = reproducer.setup_gridwide_cells(cells, active, dt)
    assert_equal(participants, 0)
    assert_true(not active[0])
    assert_true(cells[0].density_driver > 2.0e-6)
    assert_equal(cells[0].completed_steps, 0)


def test_no_active_cells_terminates_without_advancing() raises:
    var cells = make_cells(2)
    for cell in range(2):
        for n in range(reproducer.NumSpec):
            cells[cell].current.xn[n] = 0.0
        cells[cell].current.xn[2] = 1.0e30
        cells[cell].density_driver = 1.0e-10
    var active = make_mask(2)
    var result = reproducer.advance_gridwide_step(
        cells, active, 0, 0.0, 0, False
    )
    assert_equal(result.status, reproducer.Success)
    assert_equal(result.integrated_count, 0)
    assert_equal(cells[0].completed_steps, 0)
    assert_equal(cells[1].completed_steps, 0)
    assert_equal(cells[0].time, 0.0)
    assert_equal(cells[1].time, 0.0)


def test_nonzero_tile_limiter_and_cross_tile_tie() raises:
    var cells = make_cells(33)
    for cell in range(31, 33):
        for n in range(reproducer.NumSpec):
            cells[cell].current.xn[n] *= 4.0
    var active = make_mask(33)
    var proposal = reproducer.prepare_grid_timestep(
        cells, active, 0, 0.0, 0, False
    )
    assert_equal(proposal.status, reproducer.Success)
    assert_equal(proposal.active_count, 33)
    assert_equal(proposal.cell, 31)


def test_partial_tile_sizes_share_exact_output_time() raises:
    var counts = List[Int]()
    counts.append(1)
    counts.append(7)
    counts.append(8)
    counts.append(31)
    counts.append(32)
    counts.append(33)
    for count in counts:
        var cells = make_cells(count)
        var active = make_mask(count)
        var result = reproducer.advance_gridwide_step(
            cells, active, 0, 0.0, 0, False
        )
        assert_equal(result.status, reproducer.Success)
        assert_equal(result.integrated_count, count)
        for cell in range(count):
            assert_equal(cells[cell].time, result.dt)
            assert_equal(cells[cell].completed_steps, 1)


def test_common_time_and_steps() raises:
    var cells = make_cells(2)
    var active = make_mask(2)
    var result = reproducer.advance_gridwide_step(
        cells, active, 0, 0.0, 0, False
    )
    assert_equal(result.status, reproducer.Success)
    assert_equal(result.integrated_count, 2)
    assert_equal(cells[0].time, result.dt)
    assert_equal(cells[1].time, result.dt)
    assert_equal(cells[0].completed_steps, 1)
    assert_equal(cells[1].completed_steps, 1)


def test_packed_record_round_trip() raises:
    var state = reproducer.make_collapse_state()
    state.completed_steps = 17
    state.time = 123.5
    state.density_driver = 4.25e-9
    state.current.rho = 9.5
    state.current.T = 88.0
    state.current.e = -3.25
    for n in range(reproducer.NumSpec):
        state.current.xn[n] = Float64(n) + 0.125
    var record = reproducer.make_packed_final_state(state, 23, 4)
    var bytes = List[UInt8]()
    reproducer.append_packed_final_state(bytes, record)
    assert_equal(len(bytes), reproducer.PackedFinalStateBytes)
    var decoded = reproducer.decode_packed_final_state(bytes, 0)
    assert_equal(decoded.cell, 23)
    assert_equal(decoded.i, 3)
    assert_equal(decoded.j, 1)
    assert_equal(decoded.k, 1)
    assert_equal(decoded.completed_steps, 17)
    assert_equal(decoded.time, 123.5)
    assert_equal(decoded.density_driver, 4.25e-9)
    assert_equal(decoded.rho, 9.5)
    assert_equal(decoded.T, 88.0)
    assert_equal(decoded.e, -3.25)
    for n in range(reproducer.NumSpec):
        assert_equal(decoded.xn[n], Float64(n) + 0.125)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

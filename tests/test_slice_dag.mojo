# SPDX-License-Identifier: BSD-3-Clause

import reproducer
from generated.slices_jac import (
    jac_slice_0,
    jac_slice_1,
    jac_slice_2,
    jac_slice_3,
    jac_slice_4,
    jac_slice_5,
    jac_slice_6,
    jac_slice_7,
)
from generated.slices_rhs import (
    rhs_eint_slice_6,
    rhs_specie_slice_0,
    rhs_specie_slice_1,
    rhs_specie_slice_2,
    rhs_specie_slice_3,
    rhs_specie_slice_4,
    rhs_specie_slice_5,
    rhs_specie_slice_6,
    rhs_specie_slice_7,
)
from std.math import abs, max
from std.testing import TestSuite, assert_true


def assert_slice_equal(actual: Float64, expected: Float64) raises:
    if actual != actual and expected != expected:
        return
    assert_true(
        abs(actual - expected) <= 1.0e-9 * max(1.0e-300, abs(expected))
    )


def compare_slices(state: reproducer.BurnState) raises:
    var X = reproducer.SpeciesTensor.stack_allocation()
    for n in range(reproducer.NumSpec):
        reproducer.vset(X, n, state.xn[n])
    var z = reproducer.redshift()

    var expected_rhs = reproducer.VecTensor.stack_allocation()
    var sliced_rhs = reproducer.VecTensor.stack_allocation()
    reproducer.zero_vec(expected_rhs)
    reproducer.zero_vec(sliced_rhs)
    reproducer.actual_rhs(state, expected_rhs)
    rhs_specie_slice_0(state, sliced_rhs, X, z)
    rhs_specie_slice_1(state, sliced_rhs, X, z)
    rhs_specie_slice_2(state, sliced_rhs, X, z)
    rhs_specie_slice_3(state, sliced_rhs, X, z)
    rhs_specie_slice_4(state, sliced_rhs, X, z)
    rhs_specie_slice_5(state, sliced_rhs, X, z)
    rhs_specie_slice_6(state, sliced_rhs, X, z)
    rhs_specie_slice_7(state, sliced_rhs, X, z)
    reproducer.vset(
        sliced_rhs,
        reproducer.NetIenuc,
        rhs_eint_slice_6(state, X, z),
    )
    for i in range(reproducer.Neqs):
        assert_slice_equal(
            reproducer.vget(sliced_rhs, i),
            reproducer.vget(expected_rhs, i),
        )

    var expected_jac = reproducer.MatTensor.stack_allocation()
    var sliced_jac = reproducer.MatTensor.stack_allocation()
    reproducer.zero_mat(expected_jac)
    reproducer.zero_mat(sliced_jac)
    reproducer.actual_jac(state, expected_jac)
    jac_slice_0(state, sliced_jac, X, z)
    jac_slice_1(state, sliced_jac, X, z)
    jac_slice_2(state, sliced_jac, X, z)
    jac_slice_3(state, sliced_jac, X, z)
    jac_slice_4(state, sliced_jac, X, z)
    jac_slice_5(state, sliced_jac, X, z)
    jac_slice_6(state, sliced_jac, X, z)
    jac_slice_7(state, sliced_jac, X, z)
    for row in range(reproducer.Neqs):
        for column in range(reproducer.Neqs):
            assert_slice_equal(
                reproducer.mget(sliced_jac, row, column),
                reproducer.mget(expected_jac, row, column),
            )


def test_slices_match_monolithic() raises:
    for temperature in [Float64(10.0), 100.0, 1.0e4, 3.0e4]:
        var state = reproducer.make_initial_state()
        state.T = temperature
        compare_slices(state)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

# SPDX-License-Identifier: BSD-3-Clause

import reproducer
from std.testing import TestSuite, assert_equal


def test_three_way_select_handles_duplicates() raises:
    var values: List[Float64] = [5.0, 1.0, 3.0, 3.0, 9.0, 2.0]
    assert_equal(reproducer.select_kth(values, 0), 1.0)
    assert_equal(reproducer.select_kth(values, 2), 3.0)
    assert_equal(reproducer.select_kth(values, 5), 9.0)


def test_cell_summary_has_exact_even_median() raises:
    var cells = List[reproducer.CollapseState]()
    for completed in [5, 1, 3, 3]:
        var cell = reproducer.make_collapse_state()
        cell.completed_steps = completed
        cells.append(cell^)
    var summary = reproducer.summarize_cells(cells, 0)
    assert_equal(summary.minimum, 1.0)
    assert_equal(summary.median, 3.0)
    assert_equal(summary.maximum, 5.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

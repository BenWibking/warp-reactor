# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: Structured eight-warp DAG dispatch and shared-memory configuration.

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
from layout import TensorLayout, TileTensor
from std.gpu import WARP_SIZE, barrier, block_idx, thread_idx
from std.math import min


comptime NumWarps = 8
comptime TileCells = 32
comptime BatchCells = NumWarps
comptime BatchesPerTile = TileCells // BatchCells
comptime MaxSharedBytes = 48 * 1024
comptime MaxExchangeBytes = 8 * 1024
comptime GeneratedExchangeBytes = 1024


def align_up(value: Int, alignment: Int) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


struct SmemLayout:
    comptime JOffset = 0
    comptime JBytes = BatchCells * reproducer.Neqs * reproducer.Neqs * 8
    comptime BaseOffset = align_up(Self.JOffset + Self.JBytes, 16)
    comptime BaseBytes = TileCells * reproducer.Neqs * 8
    comptime CandidateOffset = align_up(Self.BaseOffset + Self.BaseBytes, 16)
    comptime CandidateBytes = TileCells * reproducer.Neqs * 8
    comptime VectorsOffset = align_up(
        Self.CandidateOffset + Self.CandidateBytes, 16
    )
    comptime VectorsBytes = 5 * BatchCells * reproducer.Neqs * 8
    comptime XtOffset = align_up(Self.VectorsOffset + Self.VectorsBytes, 16)
    comptime XtBytes = BatchCells * 16 * 8
    comptime MetadataOffset = align_up(Self.XtOffset + Self.XtBytes, 16)
    comptime MetadataBytes = 4096
    comptime ExchangeOffset = align_up(
        Self.MetadataOffset + Self.MetadataBytes, 16
    )
    comptime ExchangeBytes = GeneratedExchangeBytes
    comptime ControllerOffset = align_up(
        Self.ExchangeOffset + Self.ExchangeBytes, 16
    )
    comptime ControllerBytes = 1024
    comptime Bytes = align_up(
        Self.ControllerOffset + Self.ControllerBytes, 16
    )


struct FullBarrierSync:
    def __init__(out self):
        pass

    def phase_barrier(self):
        barrier()

    def producer_arrive(self, stage: Int):
        _ = stage
        barrier()

    def consumer_wait(self, stage: Int):
        _ = stage
        barrier()

    def consumer_release(self, stage: Int):
        _ = stage
        barrier()


def validate_config():
    comptime assert TileCells == 32
    comptime assert TileCells % BatchCells == 0
    comptime assert BatchCells == NumWarps
    comptime assert BatchCells <= WARP_SIZE
    comptime assert reproducer.Neqs <= WARP_SIZE
    comptime assert NumWarps * WARP_SIZE <= 1024
    comptime assert GeneratedExchangeBytes <= MaxExchangeBytes
    comptime assert SmemLayout.Bytes <= MaxSharedBytes


def dispatch_jac_slice(
    warp_id: Int,
    state: reproducer.BurnState,
    mut jac: reproducer.MatTensor,
    X: reproducer.SpeciesTensor,
    z: Float64,
):
    if warp_id == 0:
        jac_slice_0(state, jac, X, z)
    elif warp_id == 1:
        jac_slice_1(state, jac, X, z)
    elif warp_id == 2:
        jac_slice_2(state, jac, X, z)
    elif warp_id == 3:
        jac_slice_3(state, jac, X, z)
    elif warp_id == 4:
        jac_slice_4(state, jac, X, z)
    elif warp_id == 5:
        jac_slice_5(state, jac, X, z)
    elif warp_id == 6:
        jac_slice_6(state, jac, X, z)
    else:
        jac_slice_7(state, jac, X, z)


def dispatch_rhs_slice(
    warp_id: Int,
    state: reproducer.BurnState,
    mut rhs: reproducer.VecTensor,
    X: reproducer.SpeciesTensor,
    z: Float64,
):
    if warp_id == 0:
        rhs_specie_slice_0(state, rhs, X, z)
    elif warp_id == 1:
        rhs_specie_slice_1(state, rhs, X, z)
    elif warp_id == 2:
        rhs_specie_slice_2(state, rhs, X, z)
    elif warp_id == 3:
        rhs_specie_slice_3(state, rhs, X, z)
    elif warp_id == 4:
        rhs_specie_slice_4(state, rhs, X, z)
    elif warp_id == 5:
        rhs_specie_slice_5(state, rhs, X, z)
    elif warp_id == 6:
        rhs_specie_slice_6(state, rhs, X, z)
        reproducer.vset(rhs, reproducer.NetIenuc, rhs_eint_slice_6(state, X, z))
    else:
        rhs_specie_slice_7(state, rhs, X, z)


def dag_slice_kernel[
    StateLayout: TensorLayout,
    JacobianLayout: TensorLayout,
    RhsLayout: TensorLayout,
](
    states: TileTensor[DType.float64, StateLayout, MutAnyOrigin],
    jacobians: TileTensor[DType.float64, JacobianLayout, MutAnyOrigin],
    right_hand_sides: TileTensor[DType.float64, RhsLayout, MutAnyOrigin],
    num_cells: Int,
):
    comptime assert states.flat_rank == 2
    comptime assert jacobians.flat_rank == 2
    comptime assert right_hand_sides.flat_rank == 2
    validate_config()
    var sync = FullBarrierSync()
    var warp_id = thread_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE
    var cell = block_idx.x * BatchCells + lane
    var evaluating = warp_id < NumWarps and lane < BatchCells and cell < num_cells
    if evaluating:
        var y = reproducer.VecTensor.stack_allocation()
        for component in range(reproducer.Neqs):
            reproducer.vset(
                y,
                component,
                rebind[Scalar[DType.float64]](states[cell, component]),
            )
        var state = reproducer.burn_state_from_y(y)
        var X = reproducer.SpeciesTensor.stack_allocation()
        for species in range(reproducer.NumSpec):
            reproducer.vset(X, species, state.xn[species])
        var jac = reproducer.MatTensor.stack_allocation()
        var rhs = reproducer.VecTensor.stack_allocation()
        dispatch_jac_slice(warp_id, state, jac, X, reproducer.redshift())
        dispatch_rhs_slice(warp_id, state, rhs, X, reproducer.redshift())
        var first_row = warp_id * 2
        var last_row = min(first_row + 2, reproducer.Neqs)
        for row in range(first_row, last_row):
            for column in range(reproducer.Neqs):
                jacobians[cell, row * reproducer.Neqs + column] = (
                    reproducer.mget(jac, row, column)
                )
        var first_component = warp_id
        right_hand_sides[cell, first_component] = reproducer.vget(
            rhs, first_component
        )
        var second_component = warp_id + NumWarps
        if second_component < reproducer.NumSpec:
            right_hand_sides[cell, second_component] = reproducer.vget(
                rhs, second_component
            )
        if warp_id == 6:
            right_hand_sides[cell, reproducer.NetIenuc] = reproducer.vget(
                rhs, reproducer.NetIenuc
            )
    sync.phase_barrier()

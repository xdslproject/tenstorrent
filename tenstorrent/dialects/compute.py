from xdsl.dialects.builtin import IntegerType, Signedness, i1, MemRefType, IntegerAttr
from xdsl.ir import SSAValue, Operation, Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def, prop_def


uint8 = IntegerType(8, signedness=Signedness.UNSIGNED)
uint32 = IntegerType(32, signedness=Signedness.UNSIGNED)
uint64 = IntegerType(64, signedness=Signedness.UNSIGNED)


@irdl_op_definition
class CopyTile(IRDLOperation):
    name = 'comp.copy_tile'

    cb = operand_def(uint32)
    in_tile_index = operand_def(uint32)
    dst_tile_index = operand_def(uint32)

    def __init__(self,
                 cb: SSAValue | Operation,
                 in_tile_index: SSAValue | Operation,
                 dst_tile_index: SSAValue | Operation):
        super().__init__(operands=[cb, in_tile_index, dst_tile_index])


@irdl_op_definition
class CopyTileToDSTInitShortWithDT(IRDLOperation):
    name = 'comp.copy_tile_to_dst_init_short_with_dt'

    old_cb = operand_def(uint32)
    new_cb = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(self,
                 old_cb: SSAValue | Operation,
                 new_cb: SSAValue | Operation,
                 transpose: SSAValue | Operation):
        super().__init__(operands=[old_cb, new_cb, transpose])


@irdl_op_definition
class CopyTileToDSTInitShort(IRDLOperation):
    name = 'comp.copy_tile_to_dst_init_short'

    cb = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(self, cb: SSAValue | Operation, transpose: SSAValue | Operation):
        super().__init__(operands=[cb, transpose])


@irdl_op_definition
class CopyTileInit(IRDLOperation):
    name = 'comp.copy_tile_init'

    def __init__(self):
        super().__init__()


@irdl_op_definition
class AcquireDST(IRDLOperation):
    """
    Note: Deprecated
    """
    name = 'comp.acquire_dst'

    def __init__(self):
        super().__init__()


@irdl_op_definition
class ReleaseDST(IRDLOperation):
    name = 'comp.release_dst'


@irdl_op_definition
class TileRegsAcquire(IRDLOperation):
    name = 'comp.tile_regs_acquire'


@irdl_op_definition
class TileRegsWait(IRDLOperation):
    name = 'comp.tile_regs_wait'


@irdl_op_definition
class TileRegsCommit(IRDLOperation):
    name = 'comp.tile_regs_commit'


@irdl_op_definition
class TileRegsRelease(IRDLOperation):
    name = 'comp.tile_regs_release'


@irdl_op_definition
class AbsTileInit(IRDLOperation):
    name = 'comp.abs_tile_init'


@irdl_op_definition
class AbsTile(IRDLOperation):
    name = 'comp.abs_tile'

    idst = operand_def(uint32)

    def __init__(self, idst: SSAValue | Operation):
        super().__init__(operands=[idst])


@irdl_op_definition
class AddTilesInitNof(IRDLOperation):
    name = 'comp.add_tiles_init_nof'


@irdl_op_definition
class AddTilesInit(IRDLOperation):
    name = 'comp.add_tiles_init'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    acc_to_dest = operand_def(i1)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 acc_to_dest: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1, acc_to_dest])


@irdl_op_definition
class AddTiles(IRDLOperation):
    name = 'comp.add_tiles'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 tile0: SSAValue | Operation,
                 tile1: SSAValue | Operation,
                 dst: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst])


@irdl_op_definition
class SubTilesInitNof(IRDLOperation):
    name = 'comp.sub_tiles_init_nof'


@irdl_op_definition
class SubTilesInit(IRDLOperation):
    name = 'comp.sub_tiles_init'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    acc_to_dest = operand_def(i1)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 acc_to_dest: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1, acc_to_dest])


@irdl_op_definition
class SubTiles(IRDLOperation):
    name = 'comp.sub_tiles'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 tile0: SSAValue | Operation,
                 tile1: SSAValue | Operation,
                 dst: SSAValue | Operation):
        super().__init__(operands=[
            cb0,
            cb1,
            tile0,
            tile1,
            dst
        ])


@irdl_op_definition
class MulTilesInitF(IRDLOperation):
    name = 'comp.mul_tiles_init_f'


@irdl_op_definition
class MulTilesInit(IRDLOperation):
    name = 'comp.mul_tiles_init'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class MulTiles(IRDLOperation):
    name = 'comp.mul_tiles'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 tile0: SSAValue | Operation,
                 tile1: SSAValue | Operation,
                 dst: SSAValue | Operation):
        super().__init__(operands=[
            cb0,
            cb1,
            tile0,
            tile1,
            dst
        ])


@irdl_op_definition
class AddBcastColsInitShort(IRDLOperation):
    name = 'comp.add_bcast_cols_init_short'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class AddBcastRowsInitShort(IRDLOperation):
    name = 'comp.add_bcast_rows_init_short'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class AddTilesBcast(IRDLOperation):
    name = 'comp.add_tiles_bcast'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    # TODO: create custom BroadcastType type
    # broadcast_dimension = prop_def()

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 tile0: SSAValue | Operation,
                 tile1: SSAValue | Operation,
                 dst: SSAValue | Operation):
        super().__init__(operands=[
            cb0,
            cb1,
            tile0,
            tile1,
            dst
        ])


@irdl_op_definition
class SubBcastColsInitShort(IRDLOperation):
    name = 'comp.sub_bcast_cols_init_short'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


# TOOD: missing operation SubBcastRowsInitShort ?

@irdl_op_definition
class SubTilesBcast(IRDLOperation):
    name = 'comp.sub_tiles_bcast'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    # TODO: add BroadcastType property

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 tile0: SSAValue | Operation,
                 tile1: SSAValue | Operation,
                 dst: SSAValue | Operation):
        super().__init__(operands=[
            cb0,
            cb1,
            tile0,
            tile1,
            dst
        ])


@irdl_op_definition
class MulBcastColsInitShort(IRDLOperation):
    name = 'comp.mul_bcast_cols_init_short'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class MulBcastRowsInitShort(IRDLOperation):
    name = 'comp.mul_bcast_rows_init_short'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class MulTilesBcast(IRDLOperation):
    name = 'comp.mul_tiles_bcast'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    # TODO: add BroadcastType property

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 tile0: SSAValue | Operation,
                 tile1: SSAValue | Operation,
                 dst: SSAValue | Operation):
        super().__init__(operands=[
            cb0,
            cb1,
            tile0,
            tile1,
            dst
        ])


@irdl_op_definition
class MulTilesBcastScalarInitShort(IRDLOperation):
    name = "comp.mul_tiles_bcast_scalar_init_short"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class MulTilesBcastScalar(IRDLOperation):
    name = 'comp.mul_tiles_bcast_scalar'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 tile0: SSAValue | Operation,
                 tile1: SSAValue | Operation,
                 dst: SSAValue | Operation):
        super().__init__(operands=[
            cb0,
            cb1,
            tile0,
            tile1,
            dst
        ])


@irdl_op_definition
class MMInit(IRDLOperation):
    name = 'comp.mm_init'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 dst: SSAValue | Operation,
                 transpose: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1, dst, transpose])


@irdl_op_definition
class MMInitShortWithDT(IRDLOperation):
    name = 'comp.mm_init_short_with_dt'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 dst: SSAValue | Operation,
                 transpose: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1, dst, transpose])


@irdl_op_definition
class MMInitShort(IRDLOperation):
    name = 'comp.mm_init_short'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 dst: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1, dst])


@irdl_op_definition
class MatmulTiles(IRDLOperation):
    name = 'comp.matmul_tiles'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 tile0: SSAValue | Operation,
                 tile1: SSAValue | Operation,
                 dst: SSAValue | Operation,
                 transpose: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst, transpose])


@irdl_op_definition
class MMBlockInit(IRDLOperation):
    name = 'comp.mm_block_init'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(uint32)
    out_cols = operand_def(uint32)
    out_rows = operand_def(uint32)
    kt_dim = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 dst: SSAValue | Operation,
                 transpose: SSAValue | Operation,
                 out_cols: SSAValue | Operation,
                 out_rows: SSAValue | Operation,
                 kt_dim: SSAValue | Operation):
        super().__init__(operands=[
            cb0, cb1, dst, transpose, out_cols, out_rows, kt_dim
        ])


@irdl_op_definition
class MMBlockInitShort(IRDLOperation):
    name = 'comp.mm_block_init_short'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    transpose = operand_def(uint32)
    out_cols = operand_def(uint32)
    out_rows = operand_def(uint32)
    kt_dim = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 transpose: SSAValue | Operation,
                 out_cols: SSAValue | Operation,
                 out_rows: SSAValue | Operation,
                 kt_dim: SSAValue | Operation):
        super().__init__(operands=[
            cb0, cb1, transpose, out_cols, out_rows, kt_dim
        ])


@irdl_op_definition
class MMBlockInitShortWithDT(IRDLOperation):
    name = 'comp.mm_block_init_short_with_dt'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    prev_cb1 = operand_def(uint32)
    out_cols = operand_def(uint32)
    out_rows = operand_def(uint32)
    kt_dim = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 prev_cb1: SSAValue | Operation,
                 out_cols: SSAValue | Operation,
                 out_rows: SSAValue | Operation,
                 kt_dim: SSAValue | Operation):
        super().__init__(operands=[
            cb0, cb1, prev_cb1, out_cols, out_rows, kt_dim
        ])


@irdl_op_definition
class MatmulBlock(IRDLOperation):
    name = 'comp.matmul_block'

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(i1)
    out_cols = operand_def(uint32)
    out_rows = operand_def(uint32)
    kt_dim = operand_def(uint32)

    def __init__(self,
                 cb0: SSAValue | Operation,
                 cb1: SSAValue | Operation,
                 tile0: SSAValue | Operation,
                 tile1: SSAValue | Operation,
                 dst: SSAValue | Operation,
                 transpose: SSAValue | Operation,
                 out_cols: SSAValue | Operation,
                 out_rows: SSAValue | Operation,
                 kt_dim: SSAValue | Operation):
        super().__init__(
            operands=[
                cb0,
                cb1,
                tile0,
                tile1,
                dst,
                transpose,
                out_cols,
                out_rows,
                kt_dim
            ]
        )


@irdl_op_definition
class ExpTileInit(IRDLOperation):
    name = 'comp.exp_tile_init'

    fast_and_approx = prop_def(i1)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={
            "fast_and_approx": fast_and_approx
        })


@irdl_op_definition
class ExpTile(IRDLOperation):
    name = 'comp.exp_tile'

    fast_and_approx = prop_def(i1)

    dst = operand_def(uint32)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={
            "fast_and_approx": fast_and_approx
        })


@irdl_op_definition
class Exp2TileInit(IRDLOperation):
    name = 'comp.exp2_tile_init'


@irdl_op_definition
class Exp2Tile(IRDLOperation):
    name = 'comp.exp2_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class ExpM1TileInit(IRDLOperation):
    name = 'comp.expm1_tile_init'


@irdl_op_definition
class ExpM1Tile(IRDLOperation):
    name = 'comp.expm1_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class ReluTileInit(IRDLOperation):
    name = 'comp.relu_tile_init'


@irdl_op_definition
class ReluTile(IRDLOperation):
    name = 'comp.relu_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class ReluMaxTileInit(IRDLOperation):
    name = 'comp.relu_max_tile_init'


@irdl_op_definition
class ReluMaxTile(IRDLOperation):
    name = 'comp.relu_max_tile'

    dst = operand_def(uint32)
    upper_limit = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, upper_limit: SSAValue | Operation):
        super().__init__(operands=[dst, upper_limit])


@irdl_op_definition
class ReluMinTileInit(IRDLOperation):
    name = 'comp.relu_min_tile_init'


@irdl_op_definition
class ReluMinTile(IRDLOperation):
    name = 'comp.relu_min_tile'

    dst = operand_def(uint32)
    lower_limit = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, lower_limit: SSAValue | Operation):
        super().__init__(operands=[dst, lower_limit])


@irdl_op_definition
class LeakyReluTileInit(IRDLOperation):
    name = 'comp.leaky_relu_tile_init'

    dst = operand_def(uint32)
    slope = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, slope: SSAValue | Operation):
        super().__init__(operands=[dst, slope])


@irdl_op_definition
class EluTileInit(IRDLOperation):
    name = 'comp.elu_tile_init'


@irdl_op_definition
class EluTile(IRDLOperation):
    name = 'comp.elu_tile'

    dst = operand_def(uint32)
    slope = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, slope: SSAValue | Operation):
        super().__init__(operands=[dst, slope])


@irdl_op_definition
class ErfTileInit(IRDLOperation):
    name = 'comp.erf_tile_init'

    fast_and_approx = prop_def(i1)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={
            "fast_and_approx": fast_and_approx
        })


@irdl_op_definition
class ErfTile(IRDLOperation):
    name = 'comp.erf_tile'

    fast_and_approx = prop_def(i1)

    dst = operand_def(uint32)

    def __init__(self, fast_and_approx: IntegerAttr, dst: SSAValue | Operation):
        super().__init__(operands=[dst],
                         properties={
            "fast_and_approx": fast_and_approx
        })


@irdl_op_definition
class ErfcTileInit(IRDLOperation):
    name = 'comp.erfc_tile_init'

    fast_and_approx = prop_def(i1)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={
            "fast_and_approx": fast_and_approx
        })


@irdl_op_definition
class ErfcTile(IRDLOperation):
    name = 'comp.erfc_tile'

    fast_and_approx = prop_def(i1)

    dst = operand_def(uint32)

    def __init__(self, fast_and_approx: IntegerAttr, dst: SSAValue | Operation):
        super().__init__(
            operands=[dst],
            properties={
                "fast_and_approx": fast_and_approx
            }
        )


@irdl_op_definition
class ErfinvTileInit(IRDLOperation):
    name = 'comp.erfinv_tile_init'


@irdl_op_definition
class ErfinvTile(IRDLOperation):
    name = 'comp.erfinv_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class GeluTileInit(IRDLOperation):
    name = 'comp.gelu_tile_init'

    fast_and_approx = prop_def(i1)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={
            "fast_and_approx": fast_and_approx
        })


@irdl_op_definition
class GeluTile(IRDLOperation):
    name = 'comp.gelu_tile'

    fast_and_approx = prop_def(i1)
    dst = operand_def(uint32)

    def __init__(self, fast_and_approx: IntegerAttr, dst: SSAValue | Operation):
        super().__init__(
            operands=[dst],
            properties={
                "fast_and_approx": fast_and_approx
            }
        )


@irdl_op_definition
class HeavisideTileInit(IRDLOperation):
    name = 'comp.heaviside_tile_init'


@irdl_op_definition
class HeavisideTile(IRDLOperation):
    name = 'comp.heaviside_tile'

    param = operand_def(uint32)

    def __init__(self, param: SSAValue | Operation):
        super().__init__(operands=[param])


@irdl_op_definition
class IsInfTileInit(IRDLOperation):
    name = 'comp.isinf_tile_init'


@irdl_op_definition
class IsInfTile(IRDLOperation):
    name = 'comp.isinf_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class IsPosinfTileInit(IRDLOperation):
    name = 'comp.isposinf_tile_init'


@irdl_op_definition
class IsPosinfTile(IRDLOperation):
    name = 'comp.isposinf_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class IsNeginfTileInit(IRDLOperation):
    name = 'comp.isneginf_tile_init'


@irdl_op_definition
class IsNeginfTile(IRDLOperation):
    name = 'comp.isneginf_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class IsFiniteTileInit(IRDLOperation):
    name = 'comp.isfinite_tile_init'


@irdl_op_definition
class IsFiniteTile(IRDLOperation):
    name = 'comp.isfinite_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class IsNaNTile(IRDLOperation):
    name = 'comp.isnan_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class I0TileInit(IRDLOperation):
    name = 'comp.i0_tile_init'


@irdl_op_definition
class I0Tile(IRDLOperation):
    name = 'comp.i0_tile'

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])



Compute = Dialect(
    'comp',
    [
        CopyTile
    ],
    []
)
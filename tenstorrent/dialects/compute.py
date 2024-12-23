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
    Note Deprecated
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



Compute = Dialect(
    'comp',
    [
        CopyTile
    ],
    []
)
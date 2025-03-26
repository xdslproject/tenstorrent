from xdsl.dialects.builtin import IntegerType, Signedness, i1, IntegerAttr, i32, BoolAttr
from xdsl.ir import SSAValue, Operation, Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, prop_def

uint8 = IntegerType(8, signedness=Signedness.UNSIGNED)
uint32 = IntegerType(32, signedness=Signedness.UNSIGNED)
uint64 = IntegerType(64, signedness=Signedness.UNSIGNED)


@irdl_op_definition
class Copy(IRDLOperation):
    name = "comp.copy_tile"

    cb = operand_def(uint32)
    in_tile_index = operand_def(uint32)
    dst_tile_index = operand_def(uint32)

    def __init__(
        self,
        cb: SSAValue | Operation,
        in_tile_index: SSAValue | Operation,
        dst_tile_index: SSAValue | Operation,
    ):
        super().__init__(operands=[cb, in_tile_index, dst_tile_index])


@irdl_op_definition
class CopyToDSTInitShortWithDT(IRDLOperation):
    name = "comp.copy_tile_to_dst_init_short_with_dt"

    old_cb = operand_def(uint32)
    new_cb = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(
        self,
        old_cb: SSAValue | Operation,
        new_cb: SSAValue | Operation,
        transpose: SSAValue | Operation,
    ):
        super().__init__(operands=[old_cb, new_cb, transpose])


@irdl_op_definition
class CopyToDSTInitShort(IRDLOperation):
    name = "comp.copy_tile_to_dst_init_short"

    cb = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(self, cb: SSAValue | Operation, transpose: SSAValue | Operation):
        super().__init__(operands=[cb, transpose])


@irdl_op_definition
class CopyInit(IRDLOperation):
    name = "comp.copy_tile_init"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class AcquireDST(IRDLOperation):
    """
    Note: Deprecated
    """

    name = "comp.acquire_dst"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class ReleaseDST(IRDLOperation):
    name = "comp.release_dst"


@irdl_op_definition
class RegsAcquire(IRDLOperation):
    name = "comp.tile_regs_acquire"


@irdl_op_definition
class RegsWait(IRDLOperation):
    name = "comp.tile_regs_wait"


@irdl_op_definition
class RegsCommit(IRDLOperation):
    name = "comp.tile_regs_commit"


@irdl_op_definition
class RegsRelease(IRDLOperation):
    name = "comp.tile_regs_release"


@irdl_op_definition
class AbsInit(IRDLOperation):
    name = "comp.abs_tile_init"


@irdl_op_definition
class Abs(IRDLOperation):
    name = "comp.abs_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class AddInitNof(IRDLOperation):
    name = "comp.add_tiles_init_nof"


@irdl_op_definition
class AddInit(IRDLOperation):
    name = "comp.add_tiles_init"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    acc_to_dest = operand_def(i1)
    # TODO: default acc_to_dest == False

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        acc_to_dest: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, acc_to_dest])


@irdl_op_definition
class Add(IRDLOperation):
    name = "comp.add_tiles"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst])


@irdl_op_definition
class SubInitNof(IRDLOperation):
    name = "comp.sub_tiles_init_nof"


@irdl_op_definition
class SubInit(IRDLOperation):
    name = "comp.sub_tiles_init"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    acc_to_dest = operand_def(i1)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        acc_to_dest: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, acc_to_dest])


@irdl_op_definition
class Sub(IRDLOperation):
    name = "comp.sub_tiles"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst])


@irdl_op_definition
class MulInitF(IRDLOperation):
    name = "comp.mul_tiles_init_f"


@irdl_op_definition
class MulInit(IRDLOperation):
    name = "comp.mul_tiles_init"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class Mul(IRDLOperation):
    name = "comp.mul_tiles"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst])


@irdl_op_definition
class AddBcastColsInitShort(IRDLOperation):
    name = "comp.add_bcast_cols_init_short"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class AddBcastRowsInitShort(IRDLOperation):
    name = "comp.add_bcast_rows_init_short"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class AddBcast(IRDLOperation):
    name = "comp.add_tiles_bcast"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    # TODO: create custom BroadcastType type
    # broadcast_dimension = prop_def()

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst])


@irdl_op_definition
class SubBcastColsInitShort(IRDLOperation):
    name = "comp.sub_bcast_cols_init_short"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


# TOOD: missing operation SubBcastRowsInitShort ?


@irdl_op_definition
class SubBcast(IRDLOperation):
    name = "comp.sub_tiles_bcast"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    # TODO: add BroadcastType property

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst])


@irdl_op_definition
class MulBcastColsInitShort(IRDLOperation):
    name = "comp.mul_bcast_cols_init_short"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class MulBcastRowsInitShort(IRDLOperation):
    name = "comp.mul_bcast_rows_init_short"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class MulBcast(IRDLOperation):
    name = "comp.mul_tiles_bcast"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    # TODO: add BroadcastType property

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst])


@irdl_op_definition
class MulBcastScalarInitShort(IRDLOperation):
    name = "comp.mul_tiles_bcast_scalar_init_short"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)

    def __init__(self, cb0: SSAValue | Operation, cb1: SSAValue | Operation):
        super().__init__(operands=[cb0, cb1])


@irdl_op_definition
class MulBcastScalar(IRDLOperation):
    name = "comp.mul_tiles_bcast_scalar"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst])


@irdl_op_definition
class MMInit(IRDLOperation):
    name = "comp.mm_init"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        dst: SSAValue | Operation,
        transpose: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, dst, transpose])


@irdl_op_definition
class MMInitShortWithDT(IRDLOperation):
    name = "comp.mm_init_short_with_dt"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        dst: SSAValue | Operation,
        transpose: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, dst, transpose])


@irdl_op_definition
class MMInitShort(IRDLOperation):
    name = "comp.mm_init_short"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, dst])


@irdl_op_definition
class Matmul(IRDLOperation):
    name = "comp.matmul_tiles"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
        transpose: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst, transpose])


@irdl_op_definition
class MMBlockInit(IRDLOperation):
    name = "comp.mm_block_init"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(uint32)
    out_cols = operand_def(uint32)
    out_rows = operand_def(uint32)
    kt_dim = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        dst: SSAValue | Operation,
        transpose: SSAValue | Operation,
        out_cols: SSAValue | Operation,
        out_rows: SSAValue | Operation,
        kt_dim: SSAValue | Operation,
    ):
        super().__init__(
            operands=[cb0, cb1, dst, transpose, out_cols, out_rows, kt_dim]
        )


@irdl_op_definition
class MMBlockInitShort(IRDLOperation):
    name = "comp.mm_block_init_short"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    transpose = operand_def(uint32)
    out_cols = operand_def(uint32)
    out_rows = operand_def(uint32)
    kt_dim = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        transpose: SSAValue | Operation,
        out_cols: SSAValue | Operation,
        out_rows: SSAValue | Operation,
        kt_dim: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, transpose, out_cols, out_rows, kt_dim])


@irdl_op_definition
class MMBlockInitShortWithDT(IRDLOperation):
    name = "comp.mm_block_init_short_with_dt"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    prev_cb1 = operand_def(uint32)
    out_cols = operand_def(uint32)
    out_rows = operand_def(uint32)
    kt_dim = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        prev_cb1: SSAValue | Operation,
        out_cols: SSAValue | Operation,
        out_rows: SSAValue | Operation,
        kt_dim: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, prev_cb1, out_cols, out_rows, kt_dim])


@irdl_op_definition
class MatmulBlock(IRDLOperation):
    name = "comp.matmul_block"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)
    transpose = operand_def(i1)
    out_cols = operand_def(uint32)
    out_rows = operand_def(uint32)
    kt_dim = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
        transpose: SSAValue | Operation,
        out_cols: SSAValue | Operation,
        out_rows: SSAValue | Operation,
        kt_dim: SSAValue | Operation,
    ):
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
                kt_dim,
            ]
        )


@irdl_op_definition
class ExpInit(IRDLOperation):
    name = "comp.exp_tile_init"

    fast_and_approx = prop_def(BoolAttr)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={"fast_and_approx": fast_and_approx})


@irdl_op_definition
class Exp(IRDLOperation):
    name = "comp.exp_tile"

    fast_and_approx = prop_def(BoolAttr)

    dst = operand_def(uint32)

    def __init__(self, fast_and_approx: IntegerAttr, dst: SSAValue | Operation):
        super().__init__(
            operands=[dst], properties={"fast_and_approx": fast_and_approx}
        )


@irdl_op_definition
class Exp2Init(IRDLOperation):
    name = "comp.exp2_tile_init"


@irdl_op_definition
class Exp2(IRDLOperation):
    name = "comp.exp2_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class ExpM1Init(IRDLOperation):
    name = "comp.expm1_tile_init"


@irdl_op_definition
class ExpM1(IRDLOperation):
    name = "comp.expm1_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class ReluInit(IRDLOperation):
    name = "comp.relu_tile_init"


@irdl_op_definition
class Relu(IRDLOperation):
    name = "comp.relu_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class ReluMaxInit(IRDLOperation):
    name = "comp.relu_max_tile_init"


@irdl_op_definition
class ReluMax(IRDLOperation):
    name = "comp.relu_max_tile"

    dst = operand_def(uint32)
    upper_limit = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, upper_limit: SSAValue | Operation):
        super().__init__(operands=[dst, upper_limit])


@irdl_op_definition
class ReluMinInit(IRDLOperation):
    name = "comp.relu_min_tile_init"


@irdl_op_definition
class ReluMin(IRDLOperation):
    name = "comp.relu_min_tile"

    dst = operand_def(uint32)
    lower_limit = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, lower_limit: SSAValue | Operation):
        super().__init__(operands=[dst, lower_limit])


@irdl_op_definition
class LeakyReluInit(IRDLOperation):
    name = "comp.leaky_relu_tile_init"

    dst = operand_def(uint32)
    slope = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, slope: SSAValue | Operation):
        super().__init__(operands=[dst, slope])


@irdl_op_definition
class EluInit(IRDLOperation):
    name = "comp.elu_tile_init"


@irdl_op_definition
class Elu(IRDLOperation):
    name = "comp.elu_tile"

    dst = operand_def(uint32)
    slope = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, slope: SSAValue | Operation):
        super().__init__(operands=[dst, slope])


@irdl_op_definition
class ErfInit(IRDLOperation):
    name = "comp.erf_tile_init"

    fast_and_approx = prop_def(BoolAttr)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={"fast_and_approx": fast_and_approx})


@irdl_op_definition
class Erf(IRDLOperation):
    name = "comp.erf_tile"

    fast_and_approx = prop_def(BoolAttr)

    dst = operand_def(uint32)

    def __init__(self, fast_and_approx: IntegerAttr, dst: SSAValue | Operation):
        super().__init__(
            operands=[dst], properties={"fast_and_approx": fast_and_approx}
        )


@irdl_op_definition
class ErfcInit(IRDLOperation):
    name = "comp.erfc_tile_init"

    fast_and_approx = prop_def(BoolAttr)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={"fast_and_approx": fast_and_approx})


@irdl_op_definition
class Erfc(IRDLOperation):
    name = "comp.erfc_tile"

    fast_and_approx = prop_def(BoolAttr)

    dst = operand_def(uint32)

    def __init__(self, fast_and_approx: IntegerAttr, dst: SSAValue | Operation):
        super().__init__(
            operands=[dst], properties={"fast_and_approx": fast_and_approx}
        )


@irdl_op_definition
class ErfinvInit(IRDLOperation):
    name = "comp.erfinv_tile_init"


@irdl_op_definition
class Erfinv(IRDLOperation):
    name = "comp.erfinv_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class GeluInit(IRDLOperation):
    name = "comp.gelu_tile_init"

    fast_and_approx = prop_def(BoolAttr)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={"fast_and_approx": fast_and_approx})


@irdl_op_definition
class Gelu(IRDLOperation):
    name = "comp.gelu_tile"

    fast_and_approx = prop_def(BoolAttr)
    dst = operand_def(uint32)

    def __init__(self, fast_and_approx: IntegerAttr, dst: SSAValue | Operation):
        super().__init__(
            operands=[dst], properties={"fast_and_approx": fast_and_approx}
        )


@irdl_op_definition
class HeavisideInit(IRDLOperation):
    name = "comp.heaviside_tile_init"


@irdl_op_definition
class Heaviside(IRDLOperation):
    name = "comp.heaviside_tile"

    param = operand_def(uint32)

    def __init__(self, param: SSAValue | Operation):
        super().__init__(operands=[param])


@irdl_op_definition
class IsInfInit(IRDLOperation):
    name = "comp.isinf_tile_init"


@irdl_op_definition
class IsInf(IRDLOperation):
    name = "comp.isinf_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class IsPosinfInit(IRDLOperation):
    name = "comp.isposinf_tile_init"


@irdl_op_definition
class IsPosinf(IRDLOperation):
    name = "comp.isposinf_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class IsNeginfInit(IRDLOperation):
    name = "comp.isneginf_tile_init"


@irdl_op_definition
class IsNeginf(IRDLOperation):
    name = "comp.isneginf_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class IsFiniteInit(IRDLOperation):
    name = "comp.isfinite_tile_init"


@irdl_op_definition
class IsFinite(IRDLOperation):
    name = "comp.isfinite_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class IsNaN(IRDLOperation):
    name = "comp.isnan_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class I0Init(IRDLOperation):
    name = "comp.i0_tile_init"


@irdl_op_definition
class I0(IRDLOperation):
    name = "comp.i0_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class LogicalNotUnaryInit(IRDLOperation):
    name = "comp.logical_not_unary_tile_init"


@irdl_op_definition
class LogicalNotUnary(IRDLOperation):
    name = "comp.logical_not_unary_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class RecipInit(IRDLOperation):
    name = "comp.recip_tile_init"


@irdl_op_definition
class Recip(IRDLOperation):
    name = "comp.recip_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class SignInit(IRDLOperation):
    name = "comp.sign_tile_init"


@irdl_op_definition
class Sign(IRDLOperation):
    name = "comp.sign_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class SqrtInit(IRDLOperation):
    name = "comp.sqrt_tile_init"


@irdl_op_definition
class Sqrt(IRDLOperation):
    name = "comp.sqrt_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class RSqrtInit(IRDLOperation):
    name = "comp.rsqrt_tile_init"

    fast_and_approx = prop_def(BoolAttr)

    def __init__(self, fast_and_approx: IntegerAttr):
        super().__init__(properties={"fast_and_approx": fast_and_approx})


@irdl_op_definition
class RSqrt(IRDLOperation):
    name = "comp.rsqrt_tile"

    fast_and_approx = prop_def(BoolAttr)
    dst = operand_def(uint32)

    def __init__(self, fast_and_approx: IntegerAttr, dst: SSAValue | Operation):
        super().__init__(
            operands=[dst], properties={"fast_and_approx": fast_and_approx}
        )


@irdl_op_definition
class SigmoidInit(IRDLOperation):
    name = "comp.sigmoid_tile_init"


@irdl_op_definition
class Sigmoid(IRDLOperation):
    name = "comp.sigmoid_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class LogInit(IRDLOperation):
    name = "comp.log_tile_init"


@irdl_op_definition
class Log(IRDLOperation):
    name = "comp.log_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class LogWithBaseInit(IRDLOperation):
    name = "comp.log_with_base_tile_init"


@irdl_op_definition
class LogWithBase(IRDLOperation):
    name = "comp.log_with_base_tile"

    dst = operand_def(uint32)
    log_base = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, log_base: SSAValue | Operation):
        super().__init__(operands=[dst, log_base])


@irdl_op_definition
class PowerInit(IRDLOperation):
    name = "comp.power_tile_init"


@irdl_op_definition
class Power(IRDLOperation):
    name = "comp.power_tile"

    dst = operand_def(uint32)
    power = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, power: SSAValue | Operation):
        super().__init__(operands=[dst, power])


@irdl_op_definition
class RSubInit(IRDLOperation):
    name = "comp.rsub_tile_init"


@irdl_op_definition
class RSub(IRDLOperation):
    name = "comp.rsub_tile"

    dst = operand_def(uint32)
    param = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, param: SSAValue | Operation):
        super().__init__(operands=[dst, param])


@irdl_op_definition
class SignBitInit(IRDLOperation):
    name = "comp.signbit_tile_init"


@irdl_op_definition
class SignBit(IRDLOperation):
    name = "comp.signbit_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class SquareInit(IRDLOperation):
    name = "comp.square_tile_init"


@irdl_op_definition
class Square(IRDLOperation):
    name = "comp.square_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class Reduce(IRDLOperation):
    name = "comp.reduce_tile"

    # TODO: reduce_type = prop_def(PoolType): REDUCE_OP
    # TODO: redice_dim = prop_def(ReduceDim): REDUCE_DIM

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    tile0 = operand_def(uint32)
    tile1 = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        tile0: SSAValue | Operation,
        tile1: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, tile0, tile1, dst])


@irdl_op_definition
class TransposeWHInit(IRDLOperation):
    name = "comp.transpose_wh_tile_init"

    in_cb = operand_def(uint32)
    out_cb = operand_def(uint32)

    def __init__(self, in_cb: SSAValue | Operation, out_cb: SSAValue):
        super().__init__(operands=[in_cb, out_cb])


@irdl_op_definition
class TransposeWH(IRDLOperation):
    name = "comp.transpose_wh_tile"

    cb = operand_def(uint32)
    tile = operand_def(uint32)
    dst = operand_def(uint32)

    def __init__(
        self,
        cb: SSAValue | Operation,
        tile: SSAValue | Operation,
        dst: SSAValue | Operation,
    ):
        super().__init__(operands=[cb, tile, dst])


@irdl_op_definition
class TanhInit(IRDLOperation):
    name = "comp.tanh_tile_init"


@irdl_op_definition
class Tanh(IRDLOperation):
    name = "comp.tanh_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class TanInit(IRDLOperation):
    name = "comp.tan_tile_init"


@irdl_op_definition
class Tan(IRDLOperation):
    name = "comp.tan_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class SinInit(IRDLOperation):
    name = "comp.sin_tile_init"


@irdl_op_definition
class Sin(IRDLOperation):
    name = "comp.sin_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class CosInit(IRDLOperation):
    name = "comp.cos_tile_init"


@irdl_op_definition
class Cos(IRDLOperation):
    name = "comp.cos_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class AsinInit(IRDLOperation):
    name = "comp.asin_tile_init"


@irdl_op_definition
class Asin(IRDLOperation):
    name = "comp.asin_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class AtanInit(IRDLOperation):
    name = "comp.atan_tile_init"


@irdl_op_definition
class Atan(IRDLOperation):
    name = "comp.atan_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class AcosInit(IRDLOperation):
    name = "comp.acos_tile_init"


@irdl_op_definition
class Acos(IRDLOperation):
    name = "comp.acos_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class LtzInit(IRDLOperation):
    """
    Less than zero
    """

    name = "comp.ltz_tile_init"


@irdl_op_definition
class Ltz(IRDLOperation):
    """
    Less than zero
    """

    name = "comp.ltz_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class EqzInit(IRDLOperation):
    name = "comp.eqz_tile_init"


@irdl_op_definition
class Eqz(IRDLOperation):
    name = "comp.eqz_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class LezInit(IRDLOperation):
    name = "comp.lez_tile_init"


@irdl_op_definition
class Lez(IRDLOperation):
    name = "comp.lez_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class GtzInit(IRDLOperation):
    name = "comp.gtz_tile_init"


@irdl_op_definition
class Gtz(IRDLOperation):
    name = "comp.gtz_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class GezInit(IRDLOperation):
    name = "comp.gez_tile_init"


@irdl_op_definition
class Gez(IRDLOperation):
    name = "comp.gez_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class NezInit(IRDLOperation):
    name = "comp.nez_tile_init"


@irdl_op_definition
class Nez(IRDLOperation):
    name = "comp.nez_tile"

    dst = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation):
        super().__init__(operands=[dst])


@irdl_op_definition
class UnaryNeInit(IRDLOperation):
    name = "comp.unary_ne_tile_init"


@irdl_op_definition
class UnaryNe(IRDLOperation):
    name = "comp.unary_ne_tile"

    dst = operand_def(uint32)
    param = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, param: SSAValue | Operation):
        super().__init__(operands=[dst, param])


@irdl_op_definition
class UnaryGtInit(IRDLOperation):
    name = "comp.unary_gt_tile_init"


@irdl_op_definition
class UnaryGt(IRDLOperation):
    name = "comp.unary_gt_tile"

    dst = operand_def(uint32)
    param = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, param: SSAValue | Operation):
        super().__init__(operands=[dst, param])


@irdl_op_definition
class UnaryLtInit(IRDLOperation):
    name = "comp.unary_lt_tile_init"


@irdl_op_definition
class UnaryLt(IRDLOperation):
    name = "comp.unary_lt_tile"

    dst = operand_def(uint32)
    param = operand_def(uint32)

    def __init__(self, dst: SSAValue | Operation, param: SSAValue | Operation):
        super().__init__(operands=[dst, param])


@irdl_op_definition
class TilizeInit(IRDLOperation):
    name = "comp.tilize_init"

    in_cb = operand_def(uint32)
    block = operand_def(uint32)
    out_cb = operand_def(uint32)

    def __init__(
        self,
        in_cb: SSAValue | Operation,
        block: SSAValue | Operation,
        out_cb: SSAValue | Operation,
    ):
        super().__init__(operands=[in_cb, block, out_cb])


@irdl_op_definition
class TilizeInitShort(IRDLOperation):
    name = "comp.tilize_init_short"

    in_cb = operand_def(uint32)
    block = operand_def(uint32)
    out_cb = operand_def(uint32)

    def __init__(
        self,
        in_cb: SSAValue | Operation,
        block: SSAValue | Operation,
        out_cb: SSAValue | Operation,
    ):
        super().__init__(operands=[in_cb, block, out_cb])


@irdl_op_definition
class TilizeInitShortWithDT(IRDLOperation):
    name = "comp.tilize_init_short_with_dt"

    old_in_cb = operand_def(uint32)
    new_in_cb = operand_def(uint32)
    block = operand_def(uint32)
    out_cb = operand_def(uint32)

    def __init__(
        self,
        old_in_cb: SSAValue | Operation,
        new_in_cb: SSAValue | Operation,
        block: SSAValue | Operation,
        out_cb: SSAValue | Operation,
    ):
        super().__init__(operands=[old_in_cb, new_in_cb, block, out_cb])


@irdl_op_definition
class TilizeBlock(IRDLOperation):
    name = "comp.tilize_block"

    in_cb = operand_def(uint32)
    block = operand_def(uint32)
    out_cb = operand_def(uint32)

    def __init__(
        self,
        in_cb: SSAValue | Operation,
        block: SSAValue | Operation,
        out_cb: SSAValue | Operation,
    ):
        super().__init__(operands=[in_cb, block, out_cb])


@irdl_op_definition
class TilizeUninit(IRDLOperation):
    name = "comp.tilize_uninit"

    in_cb = operand_def(uint32)
    out_cb = operand_def(uint32)

    def __init__(self, in_cb: SSAValue | Operation, out_cb: SSAValue | Operation):
        super().__init__(operands=[in_cb, out_cb])


@irdl_op_definition
class TilizeUninitWithDT(IRDLOperation):
    name = "comp.tilize_uninit_with_dt"

    old_in_cb = operand_def(uint32)
    new_in_cb = operand_def(uint32)
    out_cb = operand_def(uint32)

    def __init__(
        self,
        old_in_cb: SSAValue | Operation,
        new_in_cb: SSAValue | Operation,
        out_cb: SSAValue | Operation,
    ):
        super().__init__(operands=[old_in_cb, new_in_cb, out_cb])


@irdl_op_definition
class UntilizeInit(IRDLOperation):
    name = "comp.untilize_init"

    in_cb = operand_def(uint32)
    out_cb = operand_def(uint32)

    def __init__(self, in_cb: SSAValue | Operation, out_cb: SSAValue | Operation):
        super().__init__(operands=[in_cb, out_cb])


@irdl_op_definition
class UntilizeInitShort(IRDLOperation):
    name = "comp.untilize_init_short"

    in_cb = operand_def(uint32)

    def __init__(self, in_cb: SSAValue | Operation):
        super().__init__(operands=[in_cb])


@irdl_op_definition
class UntilizeBlock(IRDLOperation):
    name = "comp.untilize_block"

    n = prop_def(IntegerAttr)

    in_cb = operand_def(uint32)
    block = operand_def(uint32)
    out_cb = operand_def(uint32)

    def __init__(
        self,
        n: IntegerAttr,
        in_cb: SSAValue | Operation,
        block: SSAValue | Operation,
        out_cb: SSAValue | Operation,
    ):
        super().__init__(operands=[in_cb, block, out_cb], properties={"n": n})


@irdl_op_definition
class UntilizeUninit(IRDLOperation):
    name = "comp.untilize_uninit"

    in_cb = operand_def(uint32)

    def __init__(self, in_cb: SSAValue | Operation):
        super().__init__(operands=[in_cb])


@irdl_op_definition
class BinaryOpInitCommon(IRDLOperation):
    name = "comp.binary_op_init_common"

    cb0 = operand_def(uint32)
    cb1 = operand_def(uint32)
    cb_out = operand_def(uint32)
    # TODO: cb_out should default to 16

    def __init__(
        self,
        cb0: SSAValue | Operation,
        cb1: SSAValue | Operation,
        cb_out: SSAValue | Operation,
    ):
        super().__init__(operands=[cb0, cb1, cb_out])


@irdl_op_definition
class PackTile(IRDLOperation):
    name = "comp.pack_tile"

    out_of_order_output = prop_def(IntegerAttr)

    from_dst = operand_def(uint32)
    from_cb = operand_def(uint32)
    out_tile_index = operand_def(uint32)
    # TODO: out_tile_index should default to 0

    def __init__(
        self,
        out_of_order_output: IntegerAttr,
        from_dst: SSAValue | Operation,
        from_cb: SSAValue | Operation,
        out_tile_index: SSAValue | Operation,
    ):
        super().__init__(
            operands=[from_dst, from_cb, out_tile_index],
            properties={"out_of_order_output": out_of_order_output},
        )


Compute = Dialect(
    "comp",
    [
        Copy,
        CopyToDSTInitShortWithDT,
        CopyToDSTInitShort,
        CopyInit,
        AcquireDST,
        ReleaseDST,
        RegsAcquire,
        RegsWait,
        RegsCommit,
        RegsRelease,
        BinaryOpInitCommon,
        AbsInit,
        Abs,
        AddInitNof,
        AddInit,
        Add,
        SubInitNof,
        SubInit,
        Sub,
        MulInitF,
        MulInit,
        Mul,
        AddBcastColsInitShort,
        AddBcastRowsInitShort,
        AddBcast,
        SubBcastColsInitShort,
        SubBcast,
        MulBcastColsInitShort,
        MulBcastRowsInitShort,
        MulBcast,
        MulBcastScalarInitShort,
        MulBcastScalar,
        MMInit,
        MMInitShortWithDT,
        MMInitShort,
        Matmul,
        MMBlockInit,
        MMBlockInitShort,
        MMBlockInitShortWithDT,
        MatmulBlock,
        ExpInit,
        Exp,
        Exp2Init,
        Exp2,
        ExpM1Init,
        ExpM1,
        ReluInit,
        Relu,
        ReluMaxInit,
        ReluMax,
        ReluMinInit,
        ReluMin,
        LeakyReluInit,
        EluInit,
        Elu,
        ErfInit,
        Erf,
        ErfcInit,
        Erfc,
        ErfinvInit,
        Erfinv,
        GeluInit,
        Gelu,
        HeavisideInit,
        Heaviside,
        IsInfInit,
        IsInf,
        IsPosinfInit,
        IsPosinf,
        IsNeginfInit,
        IsNeginf,
        IsFiniteInit,
        IsFinite,
        IsNaN,
        I0Init,
        I0,
        LogicalNotUnaryInit,
        LogicalNotUnary,
        RecipInit,
        Recip,
        SignInit,
        Sign,
        SqrtInit,
        Sqrt,
        RSqrtInit,
        RSqrt,
        SigmoidInit,
        Sigmoid,
        LogInit,
        Log,
        LogWithBaseInit,
        LogWithBase,
        PowerInit,
        Power,
        RSubInit,
        RSub,
        SignBitInit,
        SignBit,
        SquareInit,
        Square,
        Reduce,
        TransposeWHInit,
        TransposeWH,
        TanhInit,
        Tanh,
        TanInit,
        Tan,
        SinInit,
        Sin,
        CosInit,
        Cos,
        AsinInit,
        Asin,
        AtanInit,
        Atan,
        AcosInit,
        Acos,
        LtzInit,
        Ltz,
        EqzInit,
        Eqz,
        LezInit,
        Lez,
        GtzInit,
        Gtz,
        GezInit,
        Gez,
        NezInit,
        Nez,
        UnaryNeInit,
        UnaryNe,
        UnaryGtInit,
        UnaryGt,
        UnaryLtInit,
        UnaryLt,
        PackTile,
        TilizeInit,
        TilizeInitShort,
        TilizeInitShortWithDT,
        TilizeBlock,
        TilizeUninit,
        TilizeUninitWithDT,
        UntilizeInit,
        UntilizeInitShort,
        UntilizeBlock,
        UntilizeUninit,
    ],
    [],
)

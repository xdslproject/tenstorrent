from typing import List, Tuple

from xdsl.dialects.builtin import Operation, SSAValue, MemRefType, IndexType, IntegerType, Signedness, Float32Type
from xdsl.dialects import builtin, arith
from xdsl.utils.hints import isa

from tenstorrent.dialects import ConstExprType
from .type_checker import MLIRType



def cast_if_needed(target_type: MLIRType, ssa: SSAValue, ops) -> Tuple[List[Operation], SSAValue]:
    """
    Uses self.get_cast, but returns new values only if needed, otherwise just
    returns the old values
    """
    cast_ops, cast_ssa = get_cast(target_type, ssa)
    if cast_ssa is not None:
        return ops + cast_ops, cast_ssa

    return ops, ssa


def get_cast(target_type: MLIRType, ssa: SSAValue) -> Tuple[List[Operation], SSAValue]:
    """
    Handles conversion between two types directly.

    Also unwraps constexpr types when necessary.
    """
    found_type = ssa.type
    if target_type == found_type:
        return [], ssa

    if isinstance(target_type, ConstExprType):
        # wrap the result of a ConstExprType expression back into a ConstExprType
        if target_type.get_element_type() != ssa.type:
            raise TypeError(
                f"Attempting to cast from {ssa.type} to {target_type}"
                f" which but the target type is a constexpr"
                f" with a different element type"
            )

        wrap = builtin.UnrealizedConversionCastOp(
            operands=[ssa], result_types=[target_type]
        )
        return [wrap], wrap.results[0]

    if isinstance(found_type, ConstExprType):
        unwrap = builtin.UnrealizedConversionCastOp(
            operands=[ssa], result_types=[found_type.get_element_type()]
        )
        ops, ssa = get_cast(target_type, unwrap.results[0])
        return [unwrap] + ops, ssa

    # TODO: not strictly correct, should check elem types and lengths
    # TODO: there is an xDSL function for this... use it!
    if isa(target_type, MemRefType) and isa(ssa.type, MemRefType):
        return [], ssa

    if isinstance(ssa.type, IntegerType):
        # cast: int -> float
        if target_type == Float32Type():
            op_sign = ssa.type.signedness.data

            if op_sign in [Signedness.SIGNED, Signedness.SIGNLESS]:
                conv_op = arith.SIToFPOp(ssa, target_type)
                return [conv_op], conv_op.results[0]

            # first cast to signed, then recurse
            elif op_sign == Signedness.UNSIGNED:
                to_si = builtin.UnrealizedConversionCastOp(
                    operands=[ssa],
                    result_types=[
                        IntegerType(ssa.type.bitwidth, Signedness.SIGNED)
                    ],
                )
                ops, ssa = get_cast(target_type, to_si.results[0])
                return [to_si] + ops, ssa

        # cast: int -> index
        elif target_type == IndexType():
            conv_op = arith.IndexCastOp(ssa, IndexType())
            return [conv_op], conv_op.results[0]

        # from here casting between two integer types
        elif target_type == IntegerType:
            return [], ssa

        elif target_type.bitwidth != ssa.type.bitwidth:
            # TODO: No MLIR OP for casting downwards on bitwidths afaik
            #  if expanding: ___, if decreasing: ___
            #  also have to decide order of casting, e.g.
            #  i32 -> ui8 => (i32 -> ui32 -> ui8) or (i32 -> i8 -> ui8)
            pass

        elif target_type.bitwidth == 32 and ssa.type.bitwidth == 32:
            conv_op = builtin.UnrealizedConversionCastOp(
                operands=[ssa], result_types=[target_type]
            )
            return [conv_op], conv_op.results[0]

        else:
            raise NotImplementedError(
                f"Unsupported type cast from {ssa.type}: {target_type}"
            )

    raise NotImplementedError(
        f"Unsupported type cast {ssa.type} to {target_type}"
    )

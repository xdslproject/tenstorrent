from typing import List, Tuple

from xdsl.dialects.builtin import Operation, SSAValue, MemRefType, IndexType, IntegerType, Signedness, Float32Type, \
    ContainerType
from xdsl.dialects import builtin, arith
from xdsl.ir.core import Attribute
from xdsl.utils.type import have_compatible_shape, get_element_type_or_self

from tenstorrent.dialects import ConstExprType



def cast_if_needed(ssa: SSAValue, target_type: Attribute, ops) -> Tuple[List[Operation], SSAValue]:
    """
    Uses self.get_cast, but returns new values only if needed, otherwise just
    returns the old values
    """
    cast_ops, cast_ssa = get_cast(ssa, target_type)
    if cast_ssa is not None:
        return ops + cast_ops, cast_ssa

    return ops, ssa


def wrap_into_constexpr(ssa: SSAValue, target_type: ConstExprType) -> Tuple[List[Operation], SSAValue]:
    t1 = ssa.type
    t2 = target_type.get_element_type()
    if t1 != t2:
        raise TypeError(
            f"Attempting to cast from {t1} to {target_type}"
            f" but mismatching element types: {t1} and {t2}"
        )

    wrap = builtin.UnrealizedConversionCastOp(
        operands=[ssa], result_types=[target_type]
    )
    return [wrap], wrap.results[0]


def unwrap_from_constexpr(ssa: SSAValue, target_type) -> Tuple[List[Operation], SSAValue]:
    unwrap = builtin.UnrealizedConversionCastOp(
        operands=[ssa], result_types=[ssa.type.get_element_type()]
    )
    ops, ssa = get_cast(unwrap.results[0], target_type)
    return [unwrap] + ops, ssa


def cast_between_containers(ssa: SSAValue, target_type: ContainerType) -> Tuple[List[Operation], SSAValue]:
    if ssa.type.get_element_type() == target_type.get_element_type():
        return [], ssa

    raise TypeError(
        "Found two container types with"
        f" unimplemented cast between them: {ssa.type}, {target_type}"
    )


def cast_with_container(ssa: SSAValue, target_type: Attribute) -> Tuple[List[Operation], SSAValue]:
    found_type = ssa.type
    if isinstance(found_type, ContainerType) and isinstance(target_type, ContainerType):
        return cast_between_containers(ssa, target_type)

    if isinstance(target_type, ConstExprType):
        return wrap_into_constexpr(ssa, target_type)

    if isinstance(found_type, ConstExprType):
        return unwrap_from_constexpr(ssa, target_type)

    # handles MemRefTypes and things, after ConstExprs already been handled
    if not have_compatible_shape(ssa.type, target_type):
        raise TypeError(
            "Incompatible shapes to cast between found type:"
            f" {ssa.type}, and target type: {target_type}"
        )

    raise NotImplementedError(f"Unimplemented cast: {found_type} -> {target_type}")


def get_cast(ssa: SSAValue, target_type: Attribute) -> Tuple[List[Operation], SSAValue]:
    """
    Handles conversion between two types directly.

    Also unwraps constexpr types when necessary.
    """
    found_type = ssa.type
    if target_type == found_type:
        return [], ssa

    if isinstance(found_type, ContainerType) or isinstance(target_type, ContainerType):
        return cast_with_container(ssa, target_type)

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
                ops, ssa = get_cast(to_si.results[0], target_type)
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

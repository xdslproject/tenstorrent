from typing import Generic

from xdsl.dialects.builtin import i1, i32, IntegerType, Signedness, ContainerType
from xdsl.ir import (
    SSAValue,
    Operation,
    Dialect,
    AttributeCovT,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    operand_def,
    result_def,
    irdl_op_definition,
    ParameterDef,
)

uint32 = IntegerType(32, signedness=Signedness.UNSIGNED)


@irdl_attr_definition
class ConstExprType(
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[AttributeCovT],
):
    name = "ttshared.constexpr"

    element_type: ParameterDef[AttributeCovT]

    def __init__(
        self,
        element_type: AttributeCovT,
    ) -> None:
        super().__init__([element_type])

    def get_element_type(self) -> AttributeCovT:
        return self.element_type


@irdl_op_definition
class GetCompileTimeArgVal(IRDLOperation):
    name = "ttshared.get_compile_time_arg_val"

    # really this input should also be ConstExpr, but may
    # introduce a lot of overhead assuming most assignments
    # are const-expressible, however for future features may be required
    index = operand_def(uint32)
    result = result_def(ConstExprType(uint32))

    def __init__(self, i):
        super().__init__(operands=[i], result_types=[ConstExprType(uint32)])


TTShared = Dialect(
    "ttshared",
    [
        GetCompileTimeArgVal,
    ],
    [
        ConstExprType,
    ],
)

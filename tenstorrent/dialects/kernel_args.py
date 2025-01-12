from xdsl.dialects.builtin import IntegerType, Signedness
from xdsl.ir import SSAValue, Operation, Dialect, TypeAttribute
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def, prop_def


uint8 = IntegerType(8, signedness=Signedness.UNSIGNED)
uint32 = IntegerType(32, signedness=Signedness.UNSIGNED)
uint64 = IntegerType(64, signedness=Signedness.UNSIGNED)


@irdl_op_definition
class TTGetArgVal(IRDLOperation):
    name = "ttkernel.get_arg_val"

    T = prop_def(TypeAttribute)
    index = operand_def(uint32)

    # TODO: later use parametric datatype
    result = result_def(uint32)

    def __init__(self, dt: TypeAttribute, index: SSAValue | Operation):
        super().__init__(
            properties={"T": dt},
            operands=[index],
            result_types=[uint32]
        )


KernelArgs = Dialect(
    "ttkernel",
    [
        TTGetArgVal
    ],
    []
)
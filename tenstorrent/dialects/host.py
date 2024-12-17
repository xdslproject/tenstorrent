from xdsl.dialects.builtin import IntegerType, Signedness, i32, MemRefType
from xdsl.ir import SSAValue, Operation, Dialect, ParametrizedAttribute, TypeAttribute, OpResult
from xdsl.ir.core import Attribute
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def, irdl_attr_definition

@irdl_attr_definition
class CoreCoord(ParametrizedAttribute, TypeAttribute):
    name = "tthost.corecoord"

@irdl_op_definition
class TTHostCore(IRDLOperation):
    name = "tthost.core"

    src_noc_x = operand_def(i32)
    src_noc_y = operand_def(i32)
    res: OpResult = result_def(Attribute)

    def __init__(self,
                 src_noc_x: SSAValue | Operation,
                 src_noc_y: SSAValue | Operation):
        super().__init__(operands=[
            src_noc_x,
            src_noc_y],
          result_types=[CoreCoord()])

TTHost = Dialect(
    "tthost",
    [
        TTHostCore,
    ],
    [
        CoreCoord,
    ],
)

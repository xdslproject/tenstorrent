from xdsl.dialects.builtin import i1, i32
from xdsl.ir import SSAValue, Operation, Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def

# TODO: add traits?
# TODO: pages vs tiles?


@irdl_op_definition
class CBPagesAvailableAtFront(IRDLOperation):
    name = "cb.pages_available_at_front"

    cb_id = operand_def(i32)
    num_pages = operand_def(i32)
    result = result_def(i1)

    def __init__(self, cb_id: SSAValue | Operation, num_pages: SSAValue | Operation):
        super().__init__(operands=[cb_id, num_pages], result_types=[i1])


@irdl_op_definition
class CBWaitFront(IRDLOperation):
    name = "cb.wait_front"

    cb_id = operand_def(i32)
    num_tiles = operand_def(i32)

    def __init__(self, cb_id: SSAValue | Operation, num_tiles: SSAValue | Operation):
        super().__init__(operands=[cb_id, num_tiles])


@irdl_op_definition
class CBPagesReservableAtBack(IRDLOperation):
    name = "cb.pages_reservable_at_back"

    cb_id = operand_def(i32)
    num_pages = operand_def(i32)
    result = result_def(i1)

    def __init__(self, cb_id: SSAValue | Operation, num_pages: SSAValue | Operation):
        super().__init__(operands=[cb_id, num_pages], result_types=[i1])


@irdl_op_definition
class CBReserveBack(IRDLOperation):
    name = "cb.reserve_back"

    cb_id = operand_def(i32)
    num_tiles = operand_def(i32)

    def __init__(self, cb_id: SSAValue | Operation, num_tiles: SSAValue | Operation):
        super().__init__(operands=[cb_id, num_tiles])


@irdl_op_definition
class CBPushBack(IRDLOperation):
    name = "cb.push_back"

    cb_id = operand_def(i32)
    num_tiles = operand_def(i32)

    def __init__(self, cb_id: SSAValue | Operation, num_tiles: SSAValue | Operation):
        super().__init__(operands=[cb_id, num_tiles])


@irdl_op_definition
class CBPopFront(IRDLOperation):
    name = "cb.pop_front"

    cb_id = operand_def(i32)
    num_tiles = operand_def(i32)

    def __init__(self, cb_id: SSAValue | Operation, num_tiles: SSAValue | Operation):
        super().__init__(operands=[cb_id, num_tiles])


CircularBufferOperation = (
        CBReserveBack
        | CBPushBack
        | CBPopFront
        | CBWaitFront
        | CBPagesReservableAtBack
        | CBPagesAvailableAtFront
)


CircularBuffer = Dialect(
    "cb",
    [
        CBPagesAvailableAtFront,
        CBWaitFront,
        CBPagesReservableAtBack,
        CBReserveBack,
        CBPushBack,
        CBPopFront,
    ],
    [],
)

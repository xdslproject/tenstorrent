from xdsl.dialects.builtin import IntegerType, Signedness, i32, MemRefType
from xdsl.ir import SSAValue, Operation, Dialect, ParametrizedAttribute, TypeAttribute, OpResult
from xdsl.ir.core import Attribute
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def, irdl_attr_definition

@irdl_attr_definition
class CoreCoord(ParametrizedAttribute, TypeAttribute):
    name = "tthost.corecoord"

@irdl_attr_definition
class Buffer(ParametrizedAttribute, TypeAttribute):
    name = "tthost.buffer"

@irdl_attr_definition
class Device(ParametrizedAttribute, TypeAttribute):
    name = "tthost.device"

@irdl_attr_definition
class CommandQueue(ParametrizedAttribute, TypeAttribute):
    name = "tthost.command_queue"

@irdl_attr_definition
class DRAMBufferConfig(ParametrizedAttribute, TypeAttribute):
    name = "tthost.dram_buffer_config"

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

@irdl_op_definition
class TTCreateDevice(IRDLOperation):
    name = "tthost.create_device"

    index = operand_def(i32)
    res: OpResult = result_def(Attribute)

    def __init__(self,
                 index: SSAValue | Operation):
        super().__init__(operands=[
            index],
          result_types=[Device()])

@irdl_op_definition
class TTCreateBuffer(IRDLOperation):
    name = "tthost.create_buffer"

    config = operand_def(DRAMBufferConfig)
    res: OpResult = result_def(Attribute)

    def __init__(self,
                 config: SSAValue | Operation):
        super().__init__(operands=[
            config],
          result_types=[Buffer()])

@irdl_op_definition
class TTGetCommandQueue(IRDLOperation):
    name = "tthost.get_command_queue"

    device = operand_def(Device)
    res: OpResult = result_def(Attribute)

    def __init__(self,
                 device: SSAValue | Operation):
        super().__init__(operands=[
            device],
          result_types=[CommandQueue()])


@irdl_op_definition
class TTCreateDRAMConfig(IRDLOperation):
    name = "tthost.create_dram_configuration"

    size = operand_def(i32)
    page_size = operand_def(i32)
    res: OpResult = result_def(Attribute)

    def __init__(self,
                 size: SSAValue | Operation,
                 page_size: SSAValue | Operation):
        super().__init__(operands=[
            size,
            page_size],
          result_types=[DRAMBufferConfig()])

@irdl_op_definition
class TTEnqueueWriteBuffer(IRDLOperation):
    name = "tthost.enqueue_write_buffer"

    command_queue = operand_def(CommandQueue)
    buffer = operand_def(Buffer)
    data = operand_def(Attribute)

    def __init__(self,
                 command_queue: SSAValue | Operation,
                 buffer: SSAValue | Operation,
                 data: SSAValue | Operation):
        super().__init__(operands=[
            command_queue,
            buffer,
            data])

TTHost = Dialect(
    "tthost",
    [
        TTHostCore,
        TTCreateDevice,
        TTCreateDRAMConfig,
        TTCreateBuffer,
        TTGetCommandQueue,
        TTEnqueueWriteBuffer,
    ],
    [
        CoreCoord,
        CommandQueue,
        Buffer,
        Device,
        DRAMBufferConfig,
    ],
)

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.printer import Printer
from xdsl.utils.hints import isa

from xdsl.dialects.builtin import (
    IntegerType,
    Signedness,
    i32,
    MemRefType,
    StringAttr,
    IntAttr,
    i1,
    IndexType,
)
from xdsl.ir import (
    SSAValue,
    Operation,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
    OpResult,
    Data,
)
from xdsl.ir.core import Attribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    irdl_attr_definition,
    prop_def,
    var_operand_def,
    AttrSizedOperandSegments,
)
from enum import Enum
from xdsl.parser import AttrParser


class RISCVCoreFlags(Enum):
    DATAMOVEMENT_0 = "datamovement_0"
    DATAMOVEMENT_1 = "datamovement_1"
    COMPUTE = "compute"

    @staticmethod
    def try_parse(parser: AttrParser) -> set[RISCVCoreFlags] | None:
        for option in RISCVCoreFlags:
            if parser.parse_optional_characters(option.value) is not None:
                return {option}

        return None


@dataclass(frozen=True)
class RISCVCoreFlagsAttrBase(Data[tuple[RISCVCoreFlags, ...]]):
    @property
    def flags(self) -> set[RISCVCoreFlags]:
        """
        Returns a copy of the flags.
        """
        return set(self.data)

    def __init__(self, flags: Sequence[RISCVCoreFlags]):
        flags_: set[RISCVCoreFlags] = set(flags)

        super().__init__(tuple(flags_))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[RISCVCoreFlags, ...]:
        with parser.in_angle_brackets():
            flags = RISCVCoreFlags.try_parse(parser)
            if flags is None:
                return tuple()

            while parser.parse_optional_punctuation(",") is not None:
                flag = parser.expect(
                    lambda: RISCVCoreFlags.try_parse(parser),
                    "RISCV core variable flag expected",
                )
                flags.update(flag)

            return tuple(flags)

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            flags = self.data
            # make sure we emit flags in a consistent order
            printer.print(
                ",".join(flag.value for flag in RISCVCoreFlags if flag in flags)
            )


class MathFidelityFlags(Enum):
    LOFI = "LoFi"
    HIFI2 = "HiFi2"
    HIFI3 = "HiFi3"
    HIFI4 = "HiFi4"

    @staticmethod
    def try_parse(parser: AttrParser) -> set[MathFidelityFlags] | None:
        for option in MathFidelityFlags:
            if parser.parse_optional_characters(option.value) is not None:
                return {option}

        return None


@dataclass(frozen=True)
class MathFidelityFlagsAttrBase(Data[tuple[MathFidelityFlags, ...]]):
    @property
    def flags(self) -> set[MathFidelityFlags]:
        """
        Returns a copy of the flags.
        """
        return set(self.data)

    def __init__(self, flags: Sequence[MathFidelityFlags]):
        flags_: set[MathFidelityFlags] = set(flags)

        super().__init__(tuple(flags_))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[MathFidelityFlags, ...]:
        with parser.in_angle_brackets():
            flags = MathFidelityFlags.try_parse(parser)
            if flags is None:
                return tuple()

            while parser.parse_optional_punctuation(",") is not None:
                flag = parser.expect(
                    lambda: MathFidelityFlags.try_parse(parser),
                    "Math Fidelity variable flag expected",
                )
                flags.update(flag)

            return tuple(flags)

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            flags = self.data
            # make sure we emit flags in a consistent order
            printer.print(
                ",".join(flag.value for flag in MathFidelityFlags if flag in flags)
            )


@irdl_attr_definition
class RISCVCoreFlagsAttr(RISCVCoreFlagsAttrBase):
    name = "tthost.riscv_core"


@irdl_attr_definition
class MathFidelityFlagsAttr(MathFidelityFlagsAttrBase):
    name = "tthost.math_fidelity"


@irdl_attr_definition
class CoreCoord(ParametrizedAttribute, TypeAttribute):
    name = "tthost.corecoord"


@irdl_attr_definition
class Buffer(ParametrizedAttribute, TypeAttribute):
    name = "tthost.buffer"


@irdl_attr_definition
class Program(ParametrizedAttribute, TypeAttribute):
    name = "tthost.program"


@irdl_attr_definition
class Device(ParametrizedAttribute, TypeAttribute):
    name = "tthost.device"


@irdl_attr_definition
class Kernel(ParametrizedAttribute, TypeAttribute):
    name = "tthost.kernel"


@irdl_attr_definition
class CommandQueue(ParametrizedAttribute, TypeAttribute):
    name = "tthost.command_queue"


@irdl_attr_definition
class DRAMBufferConfig(ParametrizedAttribute, TypeAttribute):
    name = "tthost.dram_buffer_config"


@irdl_attr_definition
class CircularBufferConfig(ParametrizedAttribute, TypeAttribute):
    name = "tthost.circular_buffer_config"


@irdl_attr_definition
class CBHandle(ParametrizedAttribute, TypeAttribute):
    name = "tthost.cb_handle"


@irdl_op_definition
class TTHostCore(IRDLOperation):
    name = "tthost.core"

    src_noc_x = operand_def(i32)
    src_noc_y = operand_def(i32)
    res: OpResult = result_def(Attribute)

    def __init__(
        self, src_noc_x: SSAValue | Operation, src_noc_y: SSAValue | Operation
    ):
        super().__init__(operands=[src_noc_x, src_noc_y], result_types=[CoreCoord()])


@irdl_op_definition
class TTCreateDevice(IRDLOperation):
    name = "tthost.create_device"

    index = operand_def(i32)
    res: OpResult = result_def(Attribute)

    def __init__(self, index: SSAValue | Operation):
        super().__init__(operands=[index], result_types=[Device()])


@irdl_op_definition
class TTCreateProgram(IRDLOperation):
    name = "tthost.create_program"

    res: OpResult = result_def(Attribute)

    def __init__(self):
        super().__init__(result_types=[Program()])


@irdl_op_definition
class TTCreateBuffer(IRDLOperation):
    name = "tthost.create_buffer"

    config = operand_def(DRAMBufferConfig)
    res: OpResult = result_def(Attribute)

    def __init__(self, config: SSAValue | Operation):
        super().__init__(operands=[config], result_types=[Buffer()])


@irdl_op_definition
class TTGetCommandQueue(IRDLOperation):
    name = "tthost.get_command_queue"

    device = operand_def(Device)
    res: OpResult = result_def(Attribute)

    def __init__(self, device: SSAValue | Operation):
        super().__init__(operands=[device], result_types=[CommandQueue()])


@irdl_op_definition
class TTCreateDRAMConfig(IRDLOperation):
    name = "tthost.create_dram_configuration"

    size = operand_def(i32)
    page_size = operand_def(i32)
    res: OpResult = result_def(Attribute)

    def __init__(self, size: SSAValue | Operation, page_size: SSAValue | Operation):
        super().__init__(operands=[size, page_size], result_types=[DRAMBufferConfig()])


@irdl_op_definition
class TTCreateCBConfig(IRDLOperation):
    name = "tthost.create_cb_configuration"

    data_type = prop_def(StringAttr)
    num_buffers = operand_def(i32)
    page_size = operand_def(i32)
    cb_index = operand_def(i32)
    res: OpResult = result_def(Attribute)

    def __init__(
        self,
        num_buffers: SSAValue | Operation,
        page_size: SSAValue | Operation,
        cb_index: SSAValue | Operation,
        data_type: str | StringAttr,
    ):
        if isa(data_type, str):
            data_type = StringAttr(data_type)

        super().__init__(
            operands=[num_buffers, page_size, cb_index],
            properties={"data_type": data_type},
            result_types=[CircularBufferConfig()],
        )


@irdl_op_definition
class TTCreateCircularBuffer(IRDLOperation):
    name = "tthost.create_circular_buffer"

    program = operand_def(Program)
    core = operand_def(CoreCoord)
    config = operand_def(CircularBufferConfig)
    res: OpResult = result_def(Attribute)

    def __init__(
        self,
        program: SSAValue | Operation,
        core: SSAValue | Operation,
        config: SSAValue | Operation,
    ):
        super().__init__(operands=[program, core, config], result_types=[CBHandle()])


@irdl_op_definition
class TTCreateKernel(IRDLOperation):
    name = "tthost.create_kernel"

    program = operand_def(Program)
    core = operand_def(CoreCoord)

    kernel_name = prop_def(StringAttr)
    riscv_core = prop_def(RISCVCoreFlagsAttr)
    noc_id = prop_def(IntAttr)

    res: OpResult = result_def(Attribute)

    def __init__(
        self,
        program: SSAValue | Operation,
        core: SSAValue | Operation,
        kernel_name: str | StringAttr,
        riscv_core: RISCVCoreFlagsAttr,
        noc_id: int | IntAttr,
    ):
        if isa(kernel_name, str):
            kernel_name = StringAttr(kernel_name)

        if isa(noc_id, int):
            noc_id = IntAttr(noc_id)
        super().__init__(
            operands=[program, core],
            properties={
                "kernel_name": kernel_name,
                "riscv_core": riscv_core,
                "noc_id": noc_id,
            },
            result_types=[Kernel()],
        )


# TODO: xDSL ops with overloaded definitions?
@irdl_op_definition
class TTCreateComputeKernel(IRDLOperation):
    name = "tthost.create_compute_kernel"

    program = operand_def(Program)
    core = operand_def(CoreCoord)

    kernel_name = prop_def(StringAttr)
    riscv_core = prop_def(RISCVCoreFlagsAttr)  # implicit = Compute

    # compute config
    math_fidelity = prop_def(MathFidelityFlagsAttr)
    fp32_dest_acc_en = prop_def(Attribute)
    math_approx_mode = prop_def(Attribute)

    # TODO: should be "list of things known at compile-time"
    # compile_time_args = var_operand_def(Attribute) ?

    res: OpResult = result_def(Attribute)

    def __init__(
        self,
        program: SSAValue | Operation,
        core: SSAValue | Operation,
        kernel_name: str | StringAttr,
        math_fidelity: MathFidelityFlagsAttr,
        fp32_dest_acc_en: Attribute,
        math_approx_mode: Attribute,
    ):
        if isa(kernel_name, str):
            kernel_name = StringAttr(kernel_name)

        super().__init__(
            operands=[program, core],
            properties={
                "kernel_name": kernel_name,
                "riscv_core": RISCVCoreFlagsAttr([RISCVCoreFlags.COMPUTE]),
                "math_fidelity": math_fidelity,
                "fp32_dest_acc_en": fp32_dest_acc_en,
                "math_approx_mode": math_approx_mode,
            },
            result_types=[Kernel()],
        )


@irdl_op_definition
class TTEnqueueWriteBuffer(IRDLOperation):
    name = "tthost.enqueue_write_buffer"

    command_queue = operand_def(CommandQueue)
    buffer = operand_def(Buffer)
    data = operand_def(Attribute)
    blocking = operand_def(Attribute)

    def __init__(
        self,
        command_queue: SSAValue | Operation,
        buffer: SSAValue | Operation,
        data: SSAValue | Operation,
        blocking: SSAValue | Operation,
    ):
        super().__init__(operands=[command_queue, buffer, data, blocking])


@irdl_op_definition
class TTEnqueueReadBuffer(IRDLOperation):
    name = "tthost.enqueue_read_buffer"

    command_queue = operand_def(CommandQueue)
    buffer = operand_def(Buffer)
    data = operand_def(Attribute)
    blocking = operand_def(Attribute)

    def __init__(
        self,
        command_queue: SSAValue | Operation,
        buffer: SSAValue | Operation,
        data: SSAValue | Operation,
        blocking: SSAValue | Operation,
    ):
        super().__init__(operands=[command_queue, buffer, data, blocking])


@irdl_op_definition
class TTGetMemoryAddress(IRDLOperation):
    name = "tthost.get_memory_address"

    buffer = operand_def(Buffer)
    res: OpResult = result_def(Attribute)

    def __init__(self, buffer: SSAValue | Operation):
        super().__init__(operands=[buffer], result_types=[IndexType()])


@irdl_op_definition
class TTEnqueueProgram(IRDLOperation):
    name = "tthost.enqueue_program"

    command_queue = operand_def(CommandQueue)
    program = operand_def(Program)
    blocking = operand_def(Attribute)

    def __init__(
        self,
        command_queue: SSAValue | Operation,
        program: SSAValue | Operation,
        blocking: SSAValue | Operation,
    ):
        super().__init__(operands=[command_queue, program, blocking])


@irdl_op_definition
class TTFinish(IRDLOperation):
    name = "tthost.finish"

    command_queue = operand_def(CommandQueue)

    def __init__(self, command_queue: SSAValue | Operation):
        super().__init__(operands=[command_queue])


@irdl_op_definition
class TTCloseDevice(IRDLOperation):
    name = "tthost.close_device"

    device = operand_def(Device)

    def __init__(self, device: SSAValue | Operation):
        super().__init__(operands=[device])


@irdl_op_definition
class TTSetRuntimeArgs(IRDLOperation):
    name = "tthost.set_runtime_args"

    program = operand_def(Program)
    kernel = operand_def(Kernel)
    core = operand_def(CoreCoord)
    args = var_operand_def(Attribute)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        program: SSAValue | Operation,
        kernel: SSAValue | Operation,
        core: SSAValue | Operation,
        *args: SSAValue | Operation,
    ):
        super().__init__(operands=[program, kernel, core, list(args)])


TTHost = Dialect(
    "tthost",
    [
        TTHostCore,
        TTCreateDevice,
        TTCreateDRAMConfig,
        TTCreateCBConfig,
        TTCreateCircularBuffer,
        TTCreateBuffer,
        TTGetMemoryAddress,
        TTGetCommandQueue,
        TTEnqueueWriteBuffer,
        TTEnqueueReadBuffer,
        TTCreateProgram,
        TTCreateKernel,
        TTCreateComputeKernel,
        TTSetRuntimeArgs,
        TTEnqueueProgram,
        TTFinish,
        TTCloseDevice,
    ],
    [
        CoreCoord,
        CommandQueue,
        Buffer,
        CBHandle,
        Device,
        DRAMBufferConfig,
        CircularBufferConfig,
        Program,
        RISCVCoreFlagsAttr,
        MathFidelityFlagsAttr,
        Kernel,
    ],
)

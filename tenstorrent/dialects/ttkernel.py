from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from xdsl.dialects.builtin import MemRefType, i32, i64, IntegerAttr, FixedBitwidthType
from xdsl.ir import ParametrizedAttribute, TypeAttribute, Data, SSAValue, Operation, Dialect, Attribute
from xdsl.irdl import irdl_attr_definition, ParameterDef, irdl_op_definition, IRDLOperation, operand_def, result_def
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.hints import isa

"""
Implementations of Tenstorrent's own dialect
"""


@irdl_attr_definition
class NocAddr(ParametrizedAttribute, TypeAttribute):
    name = "ttkernel.noc_addr"  # def: TTKernel_Type<"NocAddr", "noc_addr">


class CBPortFlags(Enum):
    # TODO: these might not reflect the TT strings in the printed MLIR
    TTKernel_CBPortIn0 = "cb_in0"
    TTKernel_CBPortIn1 = "cb_in1"
    TTKernel_CBPortIn2 = "cb_in2"
    TTKernel_CBPortIn3 = "cb_in3"
    TTKernel_CBPortIn4 = "cb_in4"
    TTKernel_CBPortIn5 = "cb_in5"
    TTKernel_CBPortIn6 = "cb_in6"
    TTKernel_CBPortIn7 = "cb_in7"
    TTKernel_CBPortDataFlow0 = "cb_dataflow0"
    TTKernel_CBPortDataFlow1 = "cb_dataflow1"
    TTKernel_CBPortDataFlow2 = "cb_dataflow2"
    TTKernel_CBPortDataFlow3 = "cb_dataflow3"
    TTKernel_CBPortDataFlow4 = "cb_dataflow4"
    TTKernel_CBPortDataFlow5 = "cb_dataflow5"
    TTKernel_CBPortDataFlow6 = "cb_dataflow6"
    TTKernel_CBPortDataFlow7 = "cb_dataflow7"
    TTKernel_CBPortOut0 = "cb_out0"
    TTKernel_CBPortOut1 = "cb_out1"
    TTKernel_CBPortOut2 = "cb_out2"
    TTKernel_CBPortOut3 = "cb_out3"
    TTKernel_CBPortOut4 = "cb_out4"
    TTKernel_CBPortOut5 = "cb_out5"
    TTKernel_CBPortOut6 = "cb_out6"
    TTKernel_CBPortOut7 = "cb_out7"
    TTKernel_CBPortIntermed0 = "cb_intermediate0"
    TTKernel_CBPortIntermed1 = "cb_intermediate1"
    TTKernel_CBPortIntermed2 = "cb_intermediate2"
    TTKernel_CBPortIntermed3 = "cb_intermediate3"
    TTKernel_CBPortIntermed4 = "cb_intermediate4"
    TTKernel_CBPortIntermed5 = "cb_intermediate5"
    TTKernel_CBPortIntermed6 = "cb_intermediate6"
    TTKernel_CBPortIntermed7 = "cb_intermediate7"

    @staticmethod
    def try_parse(parser: AttrParser):
        for option in CBPortFlags:
            if parser.parse_optional_characters(option.value) is not None:
                return option

        return None
    
    
@dataclass(frozen=True)
class CBPortFlagsAttrBase(Data[tuple[CBPortFlags, ...]]):
    @property
    def flags(self) -> set[CBPortFlags]:
        """
        Returns a copy of the flags.
        """
        return set(self.data)

    def __init__(self, flags: Sequence[CBPortFlags]):
        flags_: set[CBPortFlags] = set(flags)

        super().__init__(tuple(flags_))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> CBPortFlags:
        flag = CBPortFlags.try_parse(parser)
        return flag

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            flags = self.data
            # make sure we emit flags in a consistent order
            printer.print(
                ",".join(flag.value for flag in CBPortFlags if flag in flags)
            )


@irdl_attr_definition
class CBPortAttr(CBPortFlagsAttrBase):
    name = "ttkernel.cbport"

    def print_parameter(self, printer: Printer):
        printer.print(str(self.data[0].data))


@irdl_attr_definition
class CBType(ParametrizedAttribute, TypeAttribute):
    name = "ttkernel.cb"

    cb_port: ParameterDef[CBPortAttr]
    address: ParameterDef[IntegerAttr]
    memref: ParameterDef[MemRefType]
    page_size: ParameterDef[IntegerAttr]
    num_buffers: ParameterDef[IntegerAttr]

    def __init__(self, port: CBPortAttr, address: IntegerAttr, memref: MemRefType):
        elem_type = memref.get_element_type()
        assert isinstance(elem_type, FixedBitwidthType)

        # TODO:
        #  if isa(elem_type, TileType):
        #      page_size = elem_type.size_in_bytes

        value = memref.get_shape()[-1]  * elem_type.bitwidth // 8
        page_size = IntegerAttr(value, 32)
        super().__init__([port, address, memref, page_size, IntegerAttr(1, 32)])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        parser.parse_punctuation("<", " entering CBType decl")
        cb_port = CBPortAttr([CBPortFlagsAttrBase.parse_parameter(parser)])
        parser.parse_punctuation(",", " between cb_port and address parameters")
        address = parser.parse_attribute()
        parser.parse_punctuation(",", " between address and memref parameters")
        memref_type = parser.parse_attribute()
        parser.parse_punctuation(",", " between memref_type and page_size parameters")
        page_size = parser.parse_attribute()
        parser.parse_punctuation(",", " between page_size and num_buffers parameters")
        num_buffers = parser.parse_attribute()
        parser.parse_punctuation(">", " at end of CBType attributes")

        return [cb_port, address, memref_type, page_size, num_buffers]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print(
                ", ".join(str(p) for p in self.parameters)
            )


@irdl_op_definition
class TileRegsAcquireOp(IRDLOperation):
    name = "ttkernel.tile_regs_acquire"


@irdl_op_definition
class TileRegsCommitOp(IRDLOperation):
    name = "ttkernel.tile_regs_commit"


@irdl_op_definition
class TileRegsWaitOp(IRDLOperation):
    name = "ttkernel.tile_regs_wait"


@irdl_op_definition
class TileRegsReleaseOp(IRDLOperation):
    name = "ttkernel.tile_regs_release"


@irdl_op_definition
class PackTileOp(IRDLOperation):
    name = "ttkernel.pack_tile"

    dst_index = operand_def(i32)
    out_cb = operand_def(CBType)
    out_index = operand_def(i32)

    def __init__(self, dst_index: SSAValue | Operation, out_cb: SSAValue | Operation, out_index: SSAValue | Operation):
        super().__init__(operands=[dst_index, out_cb, out_index])


@irdl_op_definition
class BinaryOpInitCommonOp(IRDLOperation):
    name = "ttkernel.binary_op_init_common"

    in0_cb = operand_def(CBType)
    in1_cb = operand_def(CBType)
    out_cb = operand_def(CBType)

    def __init__(self, in0_cb: SSAValue | Operation, in1_cb: SSAValue | Operation, out_cb: SSAValue | Operation):
        super().__init__(operands=[in0_cb, in1_cb, out_cb])


@irdl_op_definition
class AddTilesInitOp(IRDLOperation):
    name = "ttkernel.add_tiles_init"

    in0_cb = operand_def(CBType)
    in1_cb = operand_def(CBType)

    def __init__(self, in0: SSAValue | Operation, in1: SSAValue | Operation):
        super().__init__(operands=[in0, in1])


@irdl_op_definition
class AddTilesOp(IRDLOperation):
    name = "ttkernel.add_tiles"

    in0_cb = operand_def(CBType)
    in1_cb = operand_def(CBType)
    in0_tile_index = operand_def(i32)
    in1_tile_index = operand_def(i32)
    dst_index = operand_def(i32)

    def __init__(
            self,
            in0_cb: SSAValue | Operation,
            in1_cb: SSAValue | Operation,
            in0_tile_in: SSAValue | Operation,
            in1_tile_in: SSAValue | Operation,
            dst_in: SSAValue | Operation
    ):
        super().__init__(operands=[in0_cb, in1_cb, in0_tile_in, in1_tile_in, dst_in])


@irdl_op_definition
class CBWaitFrontOp(IRDLOperation):
    name = "ttkernel.cb_wait_front"

    cb = operand_def(CBType)
    num_pages = operand_def(i32)

    def __init__(self, cb: SSAValue | Operation, num_pages: SSAValue | Operation):
        super().__init__(operands=[cb, num_pages])


@irdl_op_definition
class CBPopFrontOp(IRDLOperation):
    name = "ttkernel.cb_pop_front"

    cb = operand_def(CBType)
    num_pages = operand_def(i32)

    def __init__(self, cb: SSAValue | Operation, num_pages: SSAValue | Operation):
        super().__init__(operands=[cb, num_pages])


@irdl_op_definition
class CBPushBackOp(IRDLOperation):
    name = "ttkernel.cb_push_back"

    cb = operand_def(CBType)
    num_pages = operand_def(i32)

    def __init__(self, cb: SSAValue | Operation, num_pages: SSAValue | Operation):
        super().__init__(operands=[cb, num_pages])


@irdl_op_definition
class CBReserveBackOp(IRDLOperation):
    name = "ttkernel.cb_reserve_back"

    cb = operand_def(CBType)
    num_pages = operand_def(i32)

    def __init__(self, cb: SSAValue | Operation, num_pages: SSAValue | Operation):
        super().__init__(operands=[cb, num_pages])


@irdl_op_definition
class GetNocAddrFromBankIdOp(IRDLOperation):
    name = "ttkernel.get_noc_addr_from_bank_id"

    bank_id = operand_def(i32)
    bank_address_offset = operand_def(i32)

    # TODO: implement NOCAddr
    noc_addr = result_def(NocAddr())

    def __init__(self, bank_id: SSAValue | Operation, bank_address_offset: SSAValue | Operation):
        super().__init__(operands=[bank_id, bank_address_offset], result_types=[NocAddr()])


@irdl_op_definition
class NocAsyncReadOp(IRDLOperation):
    name = "ttkernel.noc_async_read"

    src_noc_addr = operand_def(NocAddr)
    dst_local_l1_addr = operand_def(i32)
    size = operand_def(i32)

    def __init__(self, src_noc_addr: SSAValue | Operation, dst_local_l1_addr: SSAValue | Operation, size: SSAValue | Operation):
        super().__init__(operands=[src_noc_addr, dst_local_l1_addr, size])


@irdl_op_definition
class NocAsyncReadBarrierOp(IRDLOperation):
    name = "ttkernel.noc_async_read_barrier"


@irdl_op_definition
class NocAsyncWriteOp(IRDLOperation):
    name = "ttkernel.noc_async_write"

    src_local_l1_addr = operand_def(i32)
    dst_noc_addr = operand_def(NocAddr)
    size = operand_def(i32)

    def __init__(self, src_local_l1_addr: SSAValue | Operation, dst_noc_addr: SSAValue | Operation, size: SSAValue | Operation):
        super().__init__(operands=[src_local_l1_addr, dst_noc_addr, size])


@irdl_op_definition
class NocAsyncWriteBarrierOp(IRDLOperation):
    name = "ttkernel.noc_async_write_barrier"


@irdl_op_definition
class GetTileSizeOp(IRDLOperation):
    name = "ttkernel.get_tile_size"

    cb = operand_def(CBType)
    tile_size_bytes = result_def(i32)

    def __init__(self, cb: SSAValue | Operation):
        super().__init__(operands=[cb], result_types=[i32])


@irdl_op_definition
class GetReadPtrOp(IRDLOperation):
    name = "ttkernel.get_read_ptr"

    cb = operand_def(CBType)
    read_ptr = result_def(i32)

    def __init__(self, cb: SSAValue | Operation):
        super().__init__(operands=[cb], result_types=[i32])


@irdl_op_definition
class GetWritePtrOp(IRDLOperation):
    name = "ttkernel.get_write_ptr"

    cb = operand_def(CBType)
    write_ptr = result_def(i32)

    def __init__(self, cb: SSAValue | Operation):
        super().__init__(operands=[cb], result_types=[i32])


@irdl_op_definition
class GetArgValOp(IRDLOperation):
    name = "ttkernel.get_arg_val"

    arg_index = operand_def(i32)
    arg_val = result_def(i32)  # technically should include semaphore type

    def __init__(self, arg_index: SSAValue | Operation):
        super().__init__(operands=[arg_index], result_types=[i32])


TTKernel = Dialect(
    "ttkernel",
    [
        TileRegsAcquireOp,
        TileRegsWaitOp,
        TileRegsCommitOp,
        TileRegsReleaseOp,
        PackTileOp,
        BinaryOpInitCommonOp,
        AddTilesInitOp,
        AddTilesOp,
        CBWaitFrontOp,
        CBPopFrontOp,
        CBPushBackOp,
        CBReserveBackOp,
        GetNocAddrFromBankIdOp,
        NocAsyncWriteOp,
        NocAsyncReadOp,
        NocAsyncWriteBarrierOp,
        NocAsyncReadBarrierOp,
        GetTileSizeOp,
        GetReadPtrOp,
        GetWritePtrOp,
        GetArgValOp,
    ],
    [
        CBPortAttr,
        CBType,
        NocAddr,
    ],
)

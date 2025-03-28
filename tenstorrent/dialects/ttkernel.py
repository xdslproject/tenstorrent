from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from xdsl.dialects.builtin import MemRefType, i32, i64, IntegerAttr, MemRefLayoutAttr, IntAttr, FixedBitwidthType
from xdsl.ir import ParametrizedAttribute, TypeAttribute, Data, SSAValue, Operation, Dialect
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
    TTKernel_CBPortIn0 = "in0"
    TTKernel_CBPortIn1 = "in1"
    TTKernel_CBPortIn2 = "in2"
    TTKernel_CBPortIn3 = "in3"
    TTKernel_CBPortIn4 = "in4"
    TTKernel_CBPortIn5 = "in5"
    TTKernel_CBPortIn6 = "in6"
    TTKernel_CBPortIn7 = "in7"
    TTKernel_CBPortDataFlow0 = "dataflow0"
    TTKernel_CBPortDataFlow1 = "dataflow1"
    TTKernel_CBPortDataFlow2 = "dataflow2"
    TTKernel_CBPortDataFlow3 = "dataflow3"
    TTKernel_CBPortDataFlow4 = "dataflow4"
    TTKernel_CBPortDataFlow5 = "dataflow5"
    TTKernel_CBPortDataFlow6 = "dataflow6"
    TTKernel_CBPortDataFlow7 = "dataflow7"
    TTKernel_CBPortOut0 = "out0"
    TTKernel_CBPortOut1 = "out1"
    TTKernel_CBPortOut2 = "out2"
    TTKernel_CBPortOut3 = "out3"
    TTKernel_CBPortOut4 = "out4"
    TTKernel_CBPortOut5 = "out5"
    TTKernel_CBPortOut6 = "out6"
    TTKernel_CBPortOut7 = "out7"
    TTKernel_CBPortIntermed0 = "intermediate0"
    TTKernel_CBPortIntermed1 = "intermediate1"
    TTKernel_CBPortIntermed2 = "intermediate2"
    TTKernel_CBPortIntermed3 = "intermediate3"
    TTKernel_CBPortIntermed4 = "intermediate4"
    TTKernel_CBPortIntermed5 = "intermediate5"
    TTKernel_CBPortIntermed6 = "intermediate6"
    TTKernel_CBPortIntermed7 = "intermediate7"

    @staticmethod
    def try_parse(parser: AttrParser) -> set["CBPortFlags"] | None:
        for option in CBPortFlags:
            if parser.parse_optional_characters(option.value) is not None:
                return {option}

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
    def parse_parameter(cls, parser: AttrParser) -> tuple[CBPortFlags, ...]:
        with parser.in_angle_brackets():
            flags = CBPortFlags.try_parse(parser)
            if flags is None:
                return tuple()

            while parser.parse_optional_punctuation(",") is not None:
                flag = parser.expect(
                    lambda: CBPortFlags.try_parse(parser),
                    "RISCV core variable flag expected",
                )
                flags.update(flag)

            return tuple(flags)

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


@irdl_attr_definition
class CBType(ParametrizedAttribute, TypeAttribute):
    name = "ttkernel.cb"

    # cb_port: CBPort
    address: ParameterDef[IntAttr]
    memref: ParameterDef[MemRefLayoutAttr]
    page_size: ParameterDef[IntAttr]
    num_buffers: ParameterDef[IntAttr]

    # TODO: fix this
    # assembly_format = "`<` $port`,` $address`,` $memref`,` $page_size`,` $num_buffers `>`"

    def __init__(self, port: CBPortAttr, address: IntAttr, memref: MemRefType):
        elem_type = memref.get_element_type()
        assert isinstance(elem_type, FixedBitwidthType)

        # TODO:
        #  if isa(elem_type, TileType):
        #      page_size = elem_type.size_in_bytes

        page_size = IntAttr(memref.element_count() * elem_type.bitwidth / 8)
        super().__init__(port, address, memref, page_size, IntAttr(1))


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
        TileRegsCommitOp,
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

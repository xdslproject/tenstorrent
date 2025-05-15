from typing import List

from xdsl.dialects import arith
from xdsl.dialects.builtin import MemRefType, FixedBitwidthType, i1, StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.utils.hints import isa

from tenstorrent.dialects import host


def create_device_dram_buffer(host_memref: MemRefType) -> List[Operation]:
    """
    Provides the MLIR code for creating a device-side DRAM buffer using
    the Metalium API. Returns the operations needed to create the buffer
    in-order (size, dram_config, dram_buffer).
    """
    # get the buffer size
    assert isinstance(host_memref.get_element_type(), FixedBitwidthType)
    dt_size = host_memref.get_element_type().size
    buffer_size = dt_size * host_memref.element_count()
    size_op = arith.ConstantOp.from_int_and_width(buffer_size, 32)

    # create a DRAM config (needs the size above)
    dram_config_op = host.TTCreateDRAMConfig(size_op, size_op)
    dram_buffer_op = host.TTCreateBuffer(dram_config_op)

    return [size_op, dram_config_op, dram_buffer_op]


def populate_dram_buffer(
    cq: SSAValue | Operation,
    host_mem: SSAValue | Operation,
    dram_buffer: SSAValue | Operation,
    blocking: bool = False,
) -> List[Operation]:
    block = arith.ConstantOp.from_int_and_width(int(blocking), i1)
    enqueue_write = host.TTEnqueueWriteBuffer(
        cq, dram_buffer, host_mem, block
    )
    return [block, enqueue_write]


def create_circular_buffer(
        program: SSAValue | Operation,
        core: SSAValue | Operation,
        page_count: SSAValue | Operation,
        page_size: SSAValue | Operation,
        cb_idx: SSAValue | Operation,
        datatype: MemRefType | FixedBitwidthType | str | StringAttr
) -> List[Operation]:
    if not isinstance(datatype, str) and not isa(datatype, StringAttr):
        assert False

    config = host.TTCreateCBConfig(page_count, page_size, cb_idx, datatype)
    cb = host.TTCreateCircularBuffer(program, core, config)

    return [config, cb]
from typing import List

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    MemRefType,
    FixedBitwidthType,
    i1,
    StringAttr,
    i32,
    f32,
)
from xdsl.ir import Operation, SSAValue
from xdsl.utils.hints import isa

from tenstorrent.dialects import host


def create_device_dram_buffer(host_memref: MemRefType) -> List[Operation]:
    """
    Provides the MLIR code for creating a device-side DRAM buffer using
    the Metalium API. Returns the operations needed to create the buffer
    in-order (page_size, size, dram_config, dram_buffer).
    """
    # get the buffer size
    assert isinstance(host_memref.get_element_type(), FixedBitwidthType)
    dt_size = host_memref.get_element_type().size
    buffer_size = dt_size * host_memref.element_count()
    size_op = arith.ConstantOp.from_int_and_width(buffer_size, 32)

    # create a DRAM config (needs the size above)
    single_tile_size = arith.ConstantOp.from_int_and_width(32 * 32 * dt_size, 32)
    single_tile_size.results[0].name_hint = "page_size"
    dram_config_op = host.TTCreateDRAMConfig(size_op, single_tile_size)
    dram_buffer_op = host.TTCreateBuffer(dram_config_op)

    return [single_tile_size, size_op, dram_config_op, dram_buffer_op]


def populate_dram_buffer(
    cq: SSAValue | Operation,
    host_mem: SSAValue | Operation,
    dram_buffer: SSAValue | Operation,
    blocking: bool = False,
) -> List[Operation]:
    block = arith.ConstantOp.from_int_and_width(int(blocking), i1)
    enqueue_write = host.TTEnqueueWriteBuffer(cq, dram_buffer, host_mem, block)
    return [block, enqueue_write]


def create_circular_buffer(
    program: SSAValue | Operation,
    core: SSAValue | Operation,
    page_count: SSAValue | Operation,
    page_size: SSAValue | Operation,
    cb_idx: SSAValue | Operation,
    datatype: MemRefType | FixedBitwidthType | str | StringAttr,
) -> List[Operation]:
    if not isinstance(datatype, str) and not isa(datatype, StringAttr):
        assert False

    config = host.TTCreateCBConfig(page_count, page_size, cb_idx, datatype)
    cb = host.TTCreateCircularBuffer(program, core, config)

    return [config, cb]


def prepare_tensor_storage(
    program: SSAValue | Operation,
    core: SSAValue | Operation,
    cq: SSAValue | Operation,
    cb_index: int | SSAValue | Operation,
    host_mem: SSAValue,
) -> List[Operation]:
    """
    Given a standard Metalium program definition, produces MLIR, which creates
    device DRAM buffers, populates them with data asynchronously, and creates
    circular buffers.
    """
    operations = []

    assert isinstance(host_mem.type, MemRefType)
    operations += create_device_dram_buffer(host_mem.type)
    dram_buffer = operations[-1]

    operations += populate_dram_buffer(cq, host_mem, dram_buffer)

    elems = host_mem.type.element_count()
    pages = elems // 1024 + (elems % 1024 != 0)
    page_count = arith.ConstantOp.from_int_and_width(pages, 32)
    page_size = operations[0]  # reuse SSA that dram buffer uses for size

    if isinstance(cb_index, int):
        cb_index = arith.ConstantOp.from_int_and_width(cb_index, 32)
        operations += [cb_index]

    dt = host_mem.type.get_element_type()
    type_map = {i32: "int", f32: "float"}
    type_str = type_map[dt]

    operations += [page_count]
    operations += create_circular_buffer(
        program,
        core,
        page_count,
        page_size,
        cb_index,
        type_str,
    )

    return operations

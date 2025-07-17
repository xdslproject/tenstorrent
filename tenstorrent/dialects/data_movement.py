from xdsl.dialects.builtin import IntegerType, Signedness, i1, i8, i32, i64, MemRefType, IntegerAttr
from xdsl.ir import SSAValue, Operation, Dialect, Attribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    prop_def,
    opt_operand_def,
)


@irdl_op_definition
class DMNocAsyncRead(IRDLOperation):
    name = "dm.noc_async_read"

    src_noc_address = operand_def(i64)
    dst_local_l1_addr = operand_def(IntegerType | MemRefType)
    size = operand_def(i32)
    noc = opt_operand_def(i8)

    def __init__(
        self,
        src_noc_address: SSAValue | Operation,
        dst_local_l1_addr: SSAValue | Operation,
        size: SSAValue | Operation,
        noc: SSAValue | Operation = None,
    ):
        super().__init__(operands=[src_noc_address, dst_local_l1_addr, size, noc])


@irdl_op_definition
class DMNocAsyncWrite(IRDLOperation):
    name = "dm.noc_async_write"

    src_local_l1_addr = operand_def(IntegerType | MemRefType)
    dst_noc_addr = operand_def(i64)
    size = operand_def(i32)
    noc = opt_operand_def(i8)

    def __init__(
        self,
        src_local_l1_addr: SSAValue | Operation,
        dst_noc_addr: SSAValue | Operation,
        size: SSAValue | Operation,
        noc: SSAValue | Operation = None,
    ):
        super().__init__(operands=[src_local_l1_addr, dst_noc_addr, size, noc])


@irdl_op_definition
class DMNocAsyncReadBarrier(IRDLOperation):
    """
    This blocking call waits for all the outstanding enqueued ``noc_async_read``
    calls issued on the current Tensix core to complete. After returning from
    this call the ``noc_async_read`` queue will be empty for the current Tensix
    core.
    """

    name = "dm.noc_async_read_barrier"

    noc = opt_operand_def(i8)

    def __init__(self, noc: SSAValue | Operation = None):
        super().__init__(operands=[noc])


@irdl_op_definition
class DMNocAsyncWriteBarrier(IRDLOperation):
    name = "dm.noc_async_write_barrier"

    noc = opt_operand_def(i8)

    def __init__(self, noc: SSAValue | Operation = None):
        super().__init__(operands=[noc])


@irdl_op_definition
class DMNocAsyncWriteMulticast(IRDLOperation):
    name = "dm.noc_async_write_multicast"

    src_local_l1_addr = operand_def(i32)
    dst_noc_addr_multicast = operand_def(i64)
    size = operand_def(i32)
    num_dests = operand_def(i32)

    # TODO: these arguments have default values
    linked = operand_def(i1)
    multicast_path_reserve = operand_def(i1)
    noc = opt_operand_def(i8)

    def __init__(
        self,
        src_local_l1_addr: SSAValue | Operation,
        dst_noc_addr_multicast: SSAValue | Operation,
        size: SSAValue | Operation,
        num_dests: SSAValue | Operation,
        linked: SSAValue | Operation,
        multicast_path_reserve: SSAValue | Operation,
        noc: SSAValue | Operation = None,
    ):
        super().__init__(
            operands=[
                src_local_l1_addr,
                dst_noc_addr_multicast,
                size,
                num_dests,
                linked,
                multicast_path_reserve,
                noc,
            ]
        )


@irdl_op_definition
class DMNocSemaphoreSetMulticast(IRDLOperation):
    name = "dm.noc_semaphore_set_multicast"

    src_local_l1_addr = operand_def(i32)
    dst_noc_addr_multicast = operand_def(i64)
    num_dests = operand_def(i32)

    # TODO: these arguments have default values
    linked = operand_def(i1)
    multicast_path_reserve = operand_def(i1)
    noc = opt_operand_def(i8)

    def __init__(
        self,
        src_local_l1_addr: SSAValue | Operation,
        dst_noc_addr_multicast: SSAValue | Operation,
        num_dests: SSAValue | Operation,
        linked: SSAValue | Operation,
        multicast_path_reserve: SSAValue | Operation,
        noc: SSAValue | Operation = None,
    ):
        super().__init__(
            operands=[
                src_local_l1_addr,
                dst_noc_addr_multicast,
                num_dests,
                linked,
                multicast_path_reserve,
                noc,
            ]
        )


@irdl_op_definition
class DMNocSemaphoreSet(IRDLOperation):
    name = "dm.noc_semaphore_set"

    # volatile i32_t *sem_addr  : is this  memref to a i32?
    sem_addr = operand_def(MemRefType(i32, [1]))
    val = operand_def(i32)

    def __init__(self, sem_addr: SSAValue | Operation, val: SSAValue | Operation):
        super().__init__(operands=[sem_addr, val])


@irdl_op_definition
class DMNocSemaphoreWait(IRDLOperation):
    name = "dm.noc_semaphore_wait"

    # as above
    sem_addr = operand_def(MemRefType(i32, [1]))
    val = operand_def(i32)

    def __init__(self, sem_addr: SSAValue | Operation, val: SSAValue | Operation):
        super().__init__(operands=[sem_addr, val])


@irdl_op_definition
class DMNocSemaphoreInc(IRDLOperation):
    name = "dm.noc_semaphore_inc"

    addr = operand_def(i64)
    incr = operand_def(i32)
    noc_id = operand_def(i8)

    def __init__(
        self,
        addr: SSAValue | Operation,
        incr: SSAValue | Operation,
        noc_id: SSAValue | Operation,
    ):
        super().__init__(operands=[addr, incr, noc_id])


@irdl_op_definition
class DMGetNocAddrFromBankId(IRDLOperation):
    name = "dm.get_noc_addr_from_bank_id"

    dram = prop_def(IntegerAttr)

    bank_id = operand_def(i32)
    bank_address_offset = operand_def(i32)
    noc = opt_operand_def(i8)  # should have default = noc_index
    result = result_def(i64)

    def __init__(
        self,
        dram: IntegerAttr,
        bank_id: SSAValue | Operation,
        bank_address_offset: SSAValue | Operation,
        noc: SSAValue | Operation = None,
    ):
        super().__init__(
            operands=[bank_id, bank_address_offset, noc],
            properties={"dram": dram},
            result_types=[i64],
        )


@irdl_op_definition
class DMInterleavedAddrGen(IRDLOperation):
    name = "dm.interleaved_addr_gen"

    dram = prop_def(IntegerAttr)
    bank_base_address = operand_def(i32)
    page_size = operand_def(i32)

    def __init__(
        self,
        dram: IntegerAttr,
        bank_base_address: SSAValue | Operation,
        page_size: SSAValue | Operation,
    ):
        super().__init__(
            operands=[bank_base_address, page_size],
            properties={"dram": dram},
        )


DataMovement = Dialect(
    "dm",
    [
        DMNocAsyncRead,
        DMNocAsyncWrite,
        DMNocAsyncReadBarrier,
        DMNocAsyncWriteBarrier,
        DMNocAsyncWriteMulticast,
        DMNocSemaphoreSetMulticast,
        DMNocSemaphoreSet,
        DMNocSemaphoreWait,
        DMNocSemaphoreInc,
        DMGetNocAddrFromBankId,
        DMInterleavedAddrGen,
    ],
    [],
)

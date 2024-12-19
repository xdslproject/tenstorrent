from xdsl.dialects.builtin import IntegerType, Signedness, i1, MemRefType
from xdsl.ir import SSAValue, Operation, Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def


uint8 = IntegerType(8, signedness=Signedness.UNSIGNED)
uint32 = IntegerType(32, signedness=Signedness.UNSIGNED)
uint64 = IntegerType(64, signedness=Signedness.UNSIGNED)


@irdl_op_definition
class DMNocAsyncRead(IRDLOperation):
    name = "dm.noc_async_read"

    src_noc_address = operand_def(uint64)
    dst_local_l1_addr = operand_def(uint32)
    size = operand_def(uint32)
    noc = operand_def(uint8)  # TODO: default value of noc_index?

    def __init__(self,
                 src_noc_address: SSAValue | Operation,
                 dst_local_l1_addr: SSAValue | Operation,
                 size: SSAValue | Operation,
                 noc: SSAValue | Operation):
        super().__init__(operands=[
            src_noc_address,
            dst_local_l1_addr,
            size,
            noc
        ])


@irdl_op_definition
class DMNocAsyncWrite(IRDLOperation):
    name = "dm.noc_async_write"

    src_local_l1_addr = operand_def(uint32)
    dst_noc_addr = operand_def(uint64)
    size = operand_def(uint32)
    noc = operand_def(uint8)

    def __init__(self,
                 src_local_l1_addr: SSAValue | Operation,
                 dst_noc_addr: SSAValue | Operation,
                 size: SSAValue | Operation,
                 noc: SSAValue | Operation):
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

    noc = operand_def(uint8)

    def __init__(self, noc: SSAValue | Operation):
        super().__init__(operands=[noc])


@irdl_op_definition
class DMNocAsyncWriteBarrier(IRDLOperation):
    name = "dm.noc_async_write_barrier"

    noc = operand_def(uint8)

    def __init__(self, noc: SSAValue | Operation):
        super().__init__(operands=[noc])


@irdl_op_definition
class DMNocAsyncWriteMulticast(IRDLOperation):
    name = "dm.noc_async_write_multicast"

    src_local_l1_addr = operand_def(uint32)
    dst_noc_addr_multicast = operand_def(uint64)
    size = operand_def(uint32)
    num_dests = operand_def(uint32)

    # TODO: these arguments have default values
    linked = operand_def(i1)
    multicast_path_reserve = operand_def(i1)
    noc = operand_def(uint8)

    def __init__(self,
                 src_local_l1_addr: SSAValue | Operation,
                 dst_noc_addr_multicast: SSAValue | Operation,
                 size: SSAValue | Operation,
                 num_dests: SSAValue | Operation,
                 linked: SSAValue | Operation,
                 multicast_path_reserve: SSAValue | Operation,
                 noc: SSAValue | Operation):
        super().__init__(
            operands=[
                src_local_l1_addr,
                dst_noc_addr_multicast,
                size,
                num_dests,
                linked,
                multicast_path_reserve,
                noc
            ]
        )


@irdl_op_definition
class DMNocSemaphoreSetMulticast(IRDLOperation):
    name = "dm.noc_semaphore_set_multicast"

    src_local_l1_addr = operand_def(uint32)
    dst_noc_addr_multicast = operand_def(uint64)
    num_dests = operand_def(uint32)

    # TODO: these arguments have default values
    linked = operand_def(i1)
    multicast_path_reserve = operand_def(i1)
    noc = operand_def(uint8)

    def __init__(self,
                 src_local_l1_addr: SSAValue | Operation,
                 dst_noc_addr_multicast: SSAValue | Operation,
                 num_dests: SSAValue | Operation,
                 linked: SSAValue | Operation,
                 multicast_path_reserve: SSAValue | Operation,
                 noc: SSAValue | Operation):
        super().__init__(
            operands=[
                src_local_l1_addr,
                dst_noc_addr_multicast,
                num_dests,
                linked,
                multicast_path_reserve,
                noc
            ]
        )


@irdl_op_definition
class DMNocSemaphoreSet(IRDLOperation):
    name = "dm.noc_semaphore_set"

    # volatile uint32_t *sem_addr  : is this  memref to a uint32?
    sem_addr = operand_def(MemRefType(uint32, [1]))
    val = operand_def(uint32)

    def __init__(self, sem_addr: SSAValue | Operation, val: SSAValue | Operation):
        super().__init__(operands=[sem_addr, val])


@irdl_op_definition
class DMNocSemaphoreWait(IRDLOperation):
    name = "dm.noc_semaphore_wait"

    # as above
    sem_addr = operand_def(MemRefType(uint32, [1]))
    val = operand_def(uint32)

    # TODO: add MLIR validation, and use memref in place of memref.load for certain args somehow
    def __init__(self, sem_addr: SSAValue | Operation, val: SSAValue | Operation):
        super().__init__(operands=[sem_addr, val])


@irdl_op_definition
class DMNocSemaphoreInc(IRDLOperation):
    name = "dm.noc_semaphore_inc"

    addr = operand_def(uint64)
    incr = operand_def(uint32)
    noc_id = operand_def(uint8)

    def __init__(self, addr: SSAValue | Operation, incr: SSAValue | Operation, noc_id: SSAValue | Operation):
        super().__init__(operands=[addr, incr, noc_id])


@irdl_op_definition
class DMGetNocAddrFromBankId(IRDLOperation):
    name = "dm.get_noc_addr_from_bank_id"

    dram = operand_def(i1)  # TODO: really a template param in their API, prop_def

    bank_id = operand_def(uint32)
    bank_address_offset = operand_def(uint32)
    noc = operand_def(uint8)  # should have default = noc_index
    result = result_def(uint64)

    def __init__(self,
                 dram: bool,
                 bank_id: SSAValue | Operation,
                 bank_address_offset: SSAValue | Operation,
                 noc: SSAValue | Operation,
                 ):
        super().__init__(operands=[
            dram,
            bank_id,
            bank_address_offset,
            noc,
        ], result_types=[uint64])


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
    ],
    []
)

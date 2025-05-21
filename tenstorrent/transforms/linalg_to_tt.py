from typing import List, Tuple, Optional

from xdsl.context import Context
from xdsl.dialects import builtin, arith, func
from xdsl.dialects.builtin import FixedBitwidthType, BoolAttr, FunctionType
from xdsl.dialects.linalg import MatmulOp, AddOp
from xdsl.ir import Region, Block
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.rewriter import InsertPoint

from tenstorrent.dialects import *
from tenstorrent.templates import (
    prepare_tensor_storage,
)


HOST_KERNEL_NAME = "host_entry"
DATA_KERNEL_NAME = "kernel_main"
COMP_KERNEL_NAME = "MAIN"


# TODO: will need to think about non-perfect tiles, different init, different
#  params etc. What happens in Metalium?

# To add a new linalg -> TT op:
#   1. Add it to one of LINALG_TO_TT_{UNARY, BINARY}
#   2. Update get_op_args
#   3. Update get_init_args

# linalg.op -> (compute.init_op, compute.op)
LINALG_TO_TT_BINARY = {
    MatmulOp: (compute.MMInit, compute.Matmul),
    AddOp: (compute.AddInit, compute.Add),
}


# linalg.op -> (compute.init_op, compute.op)
LINALG_TO_TT_UNARY = {}


REWRITE_TYPE = MatmulOp | AddOp


class LinalgToTT(RewritePattern):
    def __init__(self):
        super().__init__()
        self._ops_replaced = 0

        self.operations_to_append = []

    def get_ops_replaced(self):
        ops_replaced = self._ops_replaced
        self._ops_replaced += 1
        return ops_replaced

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: REWRITE_TYPE, rewriter: PatternRewriter):
        self.insert_tt_call(op, rewriter)

    def insert_tt_call(self, op, rewriter: PatternRewriter):
        if type(op) not in LINALG_TO_TT_BINARY and type(op) not in LINALG_TO_TT_UNARY:
            raise NotImplementedError(f"Unhandled linalg op: {type(op)}")

        binop = type(op) in LINALG_TO_TT_BINARY

        # for now assume all linalg ops work like this, memref form
        tensor0, tensor1, tensor2 = op.operands if binop else op.operands + [None]
        t0, t1 = tensor0.type, tensor1.type

        assert isinstance(t0, MemRefType)
        assert isinstance(t1, MemRefType)
        host_code_args = (t0, t1)

        if binop:
            t2 = tensor2.type
            assert isinstance(t2, MemRefType)
            host_code_args = (t0, t1, t2)

        i = self.get_ops_replaced()
        host_f_name = HOST_KERNEL_NAME
        if not i == 0:
            host_f_name += f"_{i}"

        host_code = LinalgToTT.generate_host_code(i, *host_code_args)
        data_in_code = LinalgToTT.generate_data_in(binop)
        data_out_code = LinalgToTT.generate_data_out()
        compute_code = LinalgToTT.generate_compute(binop, op)

        arg_types = list(host_code_args)

        func_def_external = func.FuncOp.external(host_f_name, arg_types, [])
        func_call_external = func.CallOp(host_f_name, op.operands, return_types=[])

        # assumes driver code always first block
        module = op.get_toplevel_object()
        driver_block = module.body.block
        rewriter.insert_op(func_def_external, InsertPoint(driver_block))

        rewriter.replace_matched_op(
            [
                func_call_external,
            ]
        )

        self.operations_to_append += [host_code, data_in_code, compute_code, data_out_code]


    @staticmethod
    def define_kernels(program, core, i) -> List[Operation]:
        """
        returns [din, dout, comp]
        """
        suffix = f"_{i}" if i != 0 else ""

        kernel_din = host.TTCreateKernel(
            program,
            core,
            f"reader{suffix}.cpp",
            RISCVCoreFlagsAttr([RISCVCoreFlags.DATAMOVEMENT_0]),
            0,
        )
        kernel_din.results[0].name_hint = "reader_kernel"

        kernel_dout = host.TTCreateKernel(
            program,
            core,
            f"writer{suffix}.cpp",
            RISCVCoreFlagsAttr([RISCVCoreFlags.DATAMOVEMENT_1]),
            1,
        )
        kernel_dout.results[0].name_hint = "writer_kernel"

        # TODO: allow variation of fidelity, fp32_acc_dest_en, math_approx_mode
        false_attr = BoolAttr(0, i1)
        kernel_compute = host.TTCreateComputeKernel(
            program,
            core,
            f"compute{suffix}.cpp",
            MathFidelityFlagsAttr([MathFidelityFlags.LOFI]),
            false_attr,
            false_attr,
        )
        kernel_compute.results[0].name_hint = "compute_kernel"

        return [kernel_din, kernel_dout, kernel_compute]

    @staticmethod
    def generate_host_code(i: int, *memref_types: MemRefType):
        match len(memref_types):
            case 2:
                binop = False
            case 3:
                binop = True
            case _:
                raise ValueError("Expected 2 or 3 memref types")

        operations = []
        arg_types = list(memref_types)
        block = Block(arg_types=arg_types)

        # shared host-func globals
        zero = arith.ConstantOp.from_int_and_width(0, 32)
        zero.results[0].name_hint = "zero"
        one = arith.ConstantOp.from_int_and_width(1, 32)
        sixteen = arith.ConstantOp.from_int_and_width(16, 32)

        program = host.TTCreateProgram()
        program.results[0].name_hint = "prog"
        device = host.TTCreateDevice(zero)  # TODO: may later take device count as arg
        core = host.TTHostCore(zero, zero)  # TODO: update later
        cq = host.TTGetCommandQueue(device)

        setup_cb0 = prepare_tensor_storage(
            program,
            core,
            cq,
            zero,
            block.args[0],
        )

        setup_cb1 = (
            prepare_tensor_storage(
                program,
                core,
                cq,
                one,
                block.args[1],
            )
            if binop
            else []
        )

        out_array = block.args[2 if binop else 1]
        setup_cb16 = prepare_tensor_storage(
            program,
            core,
            cq,
            sixteen,
            out_array,
        )

        operations += (
            [
                zero,
                one,
                sixteen,
                program,
                device,
                core,
                cq,
            ]
            + setup_cb0
            + setup_cb1
            + setup_cb16
        )

        # prepare the arguments for each kernel
        dram_bank = zero  # TODO: generalise later, multi-core sharding, etc.
        cb_setups = (setup_cb0, setup_cb1, setup_cb16)

        sizes = [x[0] for x in cb_setups if x]
        dram_buffers = [x[2] for x in cb_setups if x]
        d_addrs = [host.TTGetMemoryAddress(b) for b in dram_buffers]

        LinalgToTT.set_name_hints("dram_addr", d_addrs)
        LinalgToTT.set_name_hints("size", sizes)

        operations += d_addrs

        # make the kernel objects
        kernels = LinalgToTT.define_kernels(program, core, i)
        operations += kernels

        set_compute_args = host.TTSetRuntimeArgs(program, kernels[2], core)

        set_dout_args = host.TTSetRuntimeArgs(
            program, kernels[1], core, *(dram_bank, d_addrs[-1], sizes[-1])
        )

        din_args = (dram_bank, dram_bank) if binop else (dram_bank,)
        din_args += tuple(d_addrs[:2]) + tuple(sizes[:2])
        set_din_args = host.TTSetRuntimeArgs(
            program,
            kernels[0],
            core,
            *din_args,
        )

        operations += [set_compute_args, set_din_args, set_dout_args]

        # launch program
        false = arith.ConstantOp.from_int_and_width(0, i1)
        launch = host.TTEnqueueProgram(cq, program, false)
        wait = host.TTFinish(cq)

        # copy the data back from device DRAM into the host array
        write_back = host.TTEnqueueReadBuffer(cq, dram_buffers[-1], out_array, false)
        close = host.TTCloseDevice(device)

        operations += [false, launch, wait, write_back, close, func.ReturnOp()]

        block.add_ops(operations)

        return builtin.ModuleOp(
            [
                func.FuncOp(
                    HOST_KERNEL_NAME + (f"_{i}" if i != 0 else ""),
                    FunctionType.from_lists(arg_types, []),
                    Region(block),
                ),
            ],
            attributes={
                "kernel_type": builtin.StringAttr("host"),
                "vis": builtin.StringAttr("external"),
            },
        )

    @staticmethod
    def set_name_hints(hint: str, ops: List[Operation]):
        for op in ops:
            op.results[0].name_hint = hint

    @staticmethod
    def generate_blocking_read(
        bank: SSAValue, addr: SSAValue, size: SSAValue, cb: int
    ) -> List[Operation]:
        bank.name_hint = "bank_id"
        addr.name_hint = "mem_addr"
        size.name_hint = "size_bytes"

        # TODO: reused values repeated for each blocking read
        true_attr = BoolAttr(1, i1)
        one = arith.ConstantOp.from_int_and_width(1, 32)
        zero_8 = arith.ConstantOp.from_int_and_width(0, 8)
        zero_ui8 = builtin.UnrealizedConversionCastOp(
            operands=[zero_8],
            result_types=[IntegerType(8, signedness=Signedness.UNSIGNED)],
        )

        cb = arith.ConstantOp.from_int_and_width(cb, 32)

        src_noc_addr = data_movement.DMGetNocAddrFromBankId(
            true_attr, bank, addr, zero_ui8
        )

        write_ptr = circular_buffer.CBGetWritePointer(cb)

        wait = circular_buffer.CBReserveBack(cb, one)
        read = data_movement.DMNocAsyncRead(src_noc_addr, write_ptr, size)
        block_op = data_movement.DMNocAsyncReadBarrier()
        push = circular_buffer.CBPushBack(cb, one)

        return [
            one,
            zero_8,
            zero_ui8,
            cb,
            src_noc_addr,
            write_ptr,
            wait,
            read,
            block_op,
            push,
        ]

    @staticmethod
    def generate_data_in(binop: bool) -> builtin.ModuleOp:
        """
        Generates a kernel which reads data from two addresses in DRAM bank 0,
        and populates CB0 and CB1 each with a single page of data.
        """
        arg_types = [uint32, uint32, uint32] * (2 if binop else 1)
        block = Block(arg_types=arg_types)

        bank_id0 = block.args[0]
        mem_addr0 = block.args[2 if binop else 1]
        size_bytes0 = block.args[4 if binop else 2]

        operations = LinalgToTT.generate_blocking_read(
            bank_id0, mem_addr0, size_bytes0, 0
        )

        if binop:
            bank_id1 = block.args[1]
            mem_addr1 = block.args[3]
            size_bytes1 = block.args[5]

            operations += LinalgToTT.generate_blocking_read(
                bank_id1, mem_addr1, size_bytes1, 1
            )

        block.add_ops(operations + [func.ReturnOp()])

        return builtin.ModuleOp(
            [
                func.FuncOp(
                    DATA_KERNEL_NAME,
                    FunctionType.from_lists(arg_types, []),
                    Region(block),
                ),
            ],
            attributes={"kernel_type": builtin.StringAttr("data_in")},
        )

    @staticmethod
    def generate_data_out() -> builtin.ModuleOp:
        """
        Generates a kernel which waits for a page to be ready in CB16, then
        consumes the page and writes it to the specified address in DRAM bank 0.
        """
        true_attr = BoolAttr(1, i1)
        arg_types = [uint32, uint32, uint32]
        block = Block(arg_types=arg_types)

        bank_id = block.args[0]
        mem_addr = block.args[1]
        size_bytes = block.args[2]

        bank_id.name_hint = "bank_id"
        mem_addr.name_hint = "mem_addr"
        size_bytes.name_hint = "size_bytes"

        dst_noc_addr = data_movement.DMGetNocAddrFromBankId(
            true_attr, bank_id, mem_addr
        )

        one = arith.ConstantOp.from_int_and_width(1, 32)
        cb16 = arith.ConstantOp.from_int_and_width(16, 32)
        l1_read_addr = circular_buffer.CBGetReadPointer(cb16)

        wait = circular_buffer.CBWaitFront(cb16, one)
        write = data_movement.DMNocAsyncWrite(l1_read_addr, dst_noc_addr, size_bytes)
        write_barrier = data_movement.DMNocAsyncWriteBarrier()
        pop = circular_buffer.CBPopFront(cb16, one)

        block.add_ops(
            [
                dst_noc_addr,
                one,
                cb16,
                l1_read_addr,
                wait,
                write,
                write_barrier,
                pop,
                func.ReturnOp(),
            ]
        )

        return builtin.ModuleOp(
            [
                func.FuncOp(
                    DATA_KERNEL_NAME,
                    FunctionType.from_lists(arg_types, []),
                    Region(block),
                )
            ],
            attributes={"kernel_type": builtin.StringAttr("data_out")},
        )

    @staticmethod
    def generate_compute(binop: bool, op: Operation) -> builtin.ModuleOp:
        zero = arith.ConstantOp.from_int_and_width(0, 32)
        one = arith.ConstantOp.from_int_and_width(1, 32)
        sixteen = arith.ConstantOp.from_int_and_width(16, 32)

        zero_u = builtin.UnrealizedConversionCastOp(
            operands=[zero], result_types=[uint32]
        )
        one_u = builtin.UnrealizedConversionCastOp(
            operands=[one], result_types=[uint32]
        )
        sixteen_u = builtin.UnrealizedConversionCastOp(
            operands=[sixteen], result_types=[uint32]
        )

        true = arith.ConstantOp.from_int_and_width(1, i1)
        false = arith.ConstantOp.from_int_and_width(0, i1)

        t = type(op)
        op_mapped = LINALG_TO_TT_BINARY[t] if binop else LINALG_TO_TT_UNARY[t]
        tt_init_op_type = op_mapped[0]
        tt_op_type = op_mapped[1]

        bin_op_cmn_init = compute.BinaryOpInitCommon(zero_u, one_u, sixteen_u)
        tt_init_op_args = LinalgToTT.get_init_args(
            tt_init_op_type, zero_u, one_u, sixteen_u, true, false
        )
        bin_op_init = tt_init_op_type(*tt_init_op_args)

        # wait for a single block of tiles in each input CB
        wait0 = circular_buffer.CBWaitFront(zero, one)
        wait1 = circular_buffer.CBWaitFront(one, one)

        # acquire 8 tile registers
        acquire_regs = compute.RegsAcquire()

        # add the first tiles in cb0 and cb1, storing the result tile in dst[0]
        tt_op_args = LinalgToTT.get_op_args(
            tt_op_type, zero_u, one_u, sixteen_u, true, false
        )
        do_tt_op = tt_op_type(*tt_op_args)

        # commit the result, signals the packer
        commit = compute.RegsCommit()
        regs_wait = compute.RegsWait()
        pack = compute.PackTile(BoolAttr(0, i1), zero_u, sixteen_u, zero_u)
        release = compute.RegsRelease()

        # tt.cb_pop_front(cb0, 1)
        cb_pop0 = circular_buffer.CBPopFront(zero, one)
        cb_pop1 = circular_buffer.CBPopFront(one, one)

        push = circular_buffer.CBPushBack(sixteen, one)

        # TODO: handle unary op init, check unary ops available on compute cores
        operations = []
        operations += [zero, one, sixteen, zero_u, one_u, sixteen_u, true, false]
        operations += [bin_op_cmn_init, bin_op_init, wait1] if binop else []
        operations += [
            wait0,
            acquire_regs,
            do_tt_op,
            commit,
            regs_wait,
            pack,
            release,
            cb_pop0,
        ]
        operations += [cb_pop1] if binop else []
        operations += [push, func.ReturnOp()]

        return builtin.ModuleOp(
            [
                func.FuncOp(
                    COMP_KERNEL_NAME,
                    FunctionType.from_lists([], []),
                    Region(Block(operations)),
                )
            ],
            attributes={"kernel_type": builtin.StringAttr("compute")},
        )

    @staticmethod
    def get_init_args(op_t, zero, one, sixteen, true, false) -> Tuple[SSAValue, ...]:
        if op_t == MMInit:
            return zero, one, sixteen, zero

        if op_t == AddInit:
            return zero, one, false

        raise NotImplementedError(f"Unhandled args for init op: {op_t.__name__}")

    @staticmethod
    def get_op_args(op_t, zero, one, sixteen, true, false) -> Tuple[SSAValue, ...]:
        if op_t == Matmul:
            return zero, one, zero, zero, zero, zero

        if op_t == Add:
            return zero, one, zero, zero, zero

        raise NotImplementedError(f"Unhandled args for op: {op_t.__name__}")


@dataclass(frozen=True)
class LinalgToTenstorrentPass(ModulePass):
    """
    This transformation takes a linalg matmul operation and rewrites it using
    the Tenstorrent xDSL dialect into host code and three Metalium kernels
    """

    name = "linalg-to-tt"

    def apply(self, ctx: Context, op: builtin.ModuleOp):
        linalg_to_tt = LinalgToTT()

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    linalg_to_tt,
                ]
            ),
            apply_recursively=False,
        )

        walker.rewrite_module(op)

        # TODO: only need to make a new container for first instance, others
        #  can reuse it somehow
        module = op.get_toplevel_object()
        container_module = builtin.ModuleOp([])

        module.regions[0].move_blocks(container_module.regions[0])

        container_module.regions[0].detach_block(0)

        # TODO: probably want to make this block construction thing more global,
        #  having this pass just return [host_code, data_in_code, ..], then
        #  later concat all these lists into [container_module, *lists] and do
        #  the transform
        block = Block(
            [container_module, *linalg_to_tt.operations_to_append]
        )

        module.regions[0].add_block(block)

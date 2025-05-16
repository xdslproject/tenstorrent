from typing import List, Tuple, Optional

from xdsl.context import MLContext
from xdsl.dialects import memref, builtin, arith, func
from xdsl.dialects.builtin import FixedBitwidthType, BoolAttr, FunctionType
from xdsl.dialects.linalg import MatmulOp
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
    create_device_dram_buffer,
    populate_dram_buffer,
    create_circular_buffer,
    prepare_tensor_storage,
)


HOST_KERNEL_NAME = "host_entry"
DATA_KERNEL_NAME = "kernel_main"
COMP_KERNEL_NAME = "MAIN"


class MatmulToTT(RewritePattern):
    def __init__(self):
        super().__init__()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter):
        self.generate_binop_code(op, rewriter)

    def generate_binop_code(self, op, rewriter: PatternRewriter):
        mat0 = op.operands[0]
        mat1 = op.operands[1]
        # TODO: owner won't always be AllocaOp
        assert isinstance(mat0.owner, memref.AllocaOp)
        assert isinstance(mat1.owner, memref.AllocaOp)

        mat2 = op.operands[2]
        assert isinstance(mat2.owner, memref.AllocaOp)

        assert isinstance(mat0.type, MemRefType)
        assert isinstance(mat1.type, MemRefType)
        assert isinstance(mat2.type, MemRefType)
        t0, t1, t2 = mat0.type, mat1.type, mat2.type

        # TODO: Each time this is called, incremement func names _i
        # TODO: Can abstract most of this out for binary operations
        # TODO: Ensure that nD-MemRefs are ok for this

        host_code = self.generate_host_code(t0, t1, t2)
        data_in_code = self.generate_data_in()
        data_out_code = self.generate_data_out()
        compute_code = self.generate_compute(True)

        arg_types = [t0, t1, t2]

        func_def_external = func.FuncOp.external(HOST_KERNEL_NAME, arg_types, [])
        func_call_external = func.CallOp(
            HOST_KERNEL_NAME, [mat0, mat1, mat2], return_types=[]
        )

        module = op.get_toplevel_object()
        top_block = module.body.block
        rewriter.insert_op(func_def_external, InsertPoint(top_block))

        rewriter.replace_matched_op(
            [
                func_call_external,
            ]
        )

        container_module = builtin.ModuleOp([])
        module.regions[0].move_blocks(container_module.regions[0])

        container_module.regions[0].detach_block(0)

        block = Block(
            [container_module, host_code, data_in_code, compute_code, data_out_code]
        )

        module.regions[0].add_block(block)

    @staticmethod
    def define_kernels(program, core) -> List[Operation]:
        """
        returns [din, dout, comp]
        """
        kernel_din = host.TTCreateKernel(
            program,
            core,
            "reader.cpp",
            RISCVCoreFlagsAttr([RISCVCoreFlags.DATAMOVEMENT_0]),
            0,
        )
        kernel_din.results[0].name_hint = "reader_kernel"

        kernel_dout = host.TTCreateKernel(
            program,
            core,
            "writer.cpp",
            RISCVCoreFlagsAttr([RISCVCoreFlags.DATAMOVEMENT_1]),
            1,
        )
        kernel_dout.results[0].name_hint = "writer_kernel"

        # TODO: allow variation of fidelity, fp32_acc_dest_en, math_approx_mode
        false_attr = BoolAttr(0, i1)
        kernel_compute = host.TTCreateComputeKernel(
            program,
            core,
            "compute.cpp",
            MathFidelityFlagsAttr([MathFidelityFlags.LOFI]),
            false_attr,
            false_attr,
        )
        kernel_compute.results[0].name_hint = "compute_kernel"

        return [kernel_din, kernel_dout, kernel_compute]

    @staticmethod
    def generate_host_code(*memref_types: MemRefType):
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
        device = host.TTCreateDevice(zero)
        core = host.TTHostCore(zero, zero)
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

        MatmulToTT.set_name_hints("dram_addr", d_addrs)
        MatmulToTT.set_name_hints("size", sizes)

        operations += d_addrs

        # make the kernel objects
        kernels = MatmulToTT.define_kernels(program, core)
        operations += kernels

        set_compute_args = host.TTSetRuntimeArgs(program, kernels[2], core)

        set_dout_args = host.TTSetRuntimeArgs(
            program, kernels[1], core, *(dram_bank, d_addrs[-1], sizes[-1])
        )

        ###################
        din_args = (dram_bank, dram_bank) if binop else (dram_bank,)
        din_args += tuple(d_addrs[:2]) + tuple(sizes[:2])
        set_din_args = host.TTSetRuntimeArgs(
            program,
            kernels[0],
            core,
            *din_args,
        )
        ###################

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
                    HOST_KERNEL_NAME,
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

    def generate_data_in(self) -> builtin.ModuleOp:
        """
        Generates a kernel which reads data from two addresses in DRAM bank 0,
        and populates CB0 and CB1 each with a single page of data.
        """
        arg_types = [uint32, uint32, uint32, uint32, uint32, uint32]
        block = Block(arg_types=arg_types)

        bank_id0 = block.args[0]
        bank_id1 = block.args[1]
        mem_addr0 = block.args[2]
        mem_addr1 = block.args[3]
        size_bytes0 = block.args[4]
        size_bytes1 = block.args[5]

        bank_id0.name_hint = "bank_id0"
        bank_id1.name_hint = "bank_id1"
        mem_addr0.name_hint = "mem_addr0"
        mem_addr1.name_hint = "mem_addr1"
        size_bytes0.name_hint = "size_bytes0"
        size_bytes1.name_hint = "size_bytes1"

        operations = []

        zero_8 = arith.ConstantOp.from_int_and_width(0, 8)
        zero_ui8 = builtin.UnrealizedConversionCastOp(
            operands=[zero_8],
            result_types=[IntegerType(8, signedness=Signedness.UNSIGNED)],
        )
        operations += [zero_8, zero_ui8]

        true_attr = BoolAttr(1, i1)

        src0_noc_addr = data_movement.DMGetNocAddrFromBankId(
            true_attr, bank_id0, mem_addr0, zero_ui8
        )
        src1_noc_addr = data_movement.DMGetNocAddrFromBankId(
            true_attr, bank_id1, mem_addr1, zero_ui8
        )

        zero = arith.ConstantOp.from_int_and_width(0, 32)
        one = arith.ConstantOp.from_int_and_width(1, 32)
        cb0 = zero
        cb1 = one

        wp0 = circular_buffer.CBGetWritePointer(cb0)
        wp1 = circular_buffer.CBGetWritePointer(cb1)

        operations += [zero, one, src0_noc_addr, src1_noc_addr, wp0, wp1]

        indexed_args = [
            (cb0, size_bytes0, src0_noc_addr, wp0),
            (cb1, size_bytes1, src1_noc_addr, wp1),
        ]

        for cb, size, noc_addr, wp in indexed_args:
            wait = circular_buffer.CBReserveBack(cb, one)
            read = data_movement.DMNocAsyncRead(noc_addr, wp, size)
            block_read = (
                data_movement.DMNocAsyncReadBarrier()
            )  # TODO: necessary for twice?
            push = circular_buffer.CBPushBack(cb, one)
            operations += [wait, read, block_read, push]

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

    def generate_data_out(self) -> builtin.ModuleOp:
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

    def generate_compute(self, binop: bool) -> builtin.ModuleOp:
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

        bin_op_cmn_init = compute.BinaryOpInitCommon(zero_u, one_u, sixteen_u)
        bin_op_init = compute.MMInit(zero_u, one_u, zero_u, zero_u)

        # wait for a single block of tiles in each input CB
        wait0 = circular_buffer.CBWaitFront(zero, one)
        wait1 = circular_buffer.CBWaitFront(one, one)

        # acquire 8 tile registers
        acquire_regs = compute.RegsAcquire()

        # add the first tiles in cb0 and cb1, storing the result tile
        do_matmul = compute.Matmul(zero_u, one_u, zero_u, zero_u, zero_u, zero_u)

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
        operations += [zero, one, sixteen, zero_u, one_u, sixteen_u]
        operations += [bin_op_cmn_init, bin_op_init, wait1] if binop else []
        operations += [
            wait0,
            acquire_regs,
            do_matmul,
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


@dataclass(frozen=True)
class RewriteMatmulToTT(ModulePass):
    """
    This transformation takes a linalg matmul operation and rewrites it using
    the Tenstorrent xDSL dialect into host code and three Metalium kernels
    """

    name = "rewrite-matmul-to-tt"

    def apply(self, ctx: MLContext, input_module: builtin.ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    MatmulToTT(),
                ]
            ),
            apply_recursively=False,
        )

        walker.rewrite_module(input_module)

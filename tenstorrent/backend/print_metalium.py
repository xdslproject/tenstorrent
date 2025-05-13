from datetime import datetime as dt
import os

from xdsl.dialects.builtin import Signedness
from xdsl.ir import (
    Block,
    Region,
    Operation,
    SSAValue,
)

from xdsl.utils.hints import isa

import xdsl.dialects.arith as arith
import xdsl.dialects.memref as memref
import xdsl.dialects.scf as scf
import xdsl.dialects.func as func
import xdsl.dialects.builtin as builtin
import xdsl.dialects.printf as printf

import tenstorrent.dialects.host as host
import tenstorrent.dialects.circular_buffer as circular_buffer
import tenstorrent.dialects.data_movement as data_movement
import tenstorrent.dialects.compute as compute
import tenstorrent.dialects.ttshared as ttshared


ArithmeticOperation = (
    arith.SignlessIntegerBinaryOperation | arith.FloatingPointLikeBinaryOperation
)
BooleanOperation = arith.AndIOp | arith.OrIOp | arith.CmpiOp | arith.CmpfOp
BinaryOperation = ArithmeticOperation | BooleanOperation
OpWithBody = func.FuncOp | scf.ForOp | scf.WhileOp
CircularBufferOperationWithResult = (
    circular_buffer.CBPagesAvailableAtFront | circular_buffer.CBPagesReservableAtBack
)

TRUE = builtin.IntegerAttr.from_int_and_width(1, 1)

TenstorrentStmts = [
    data_movement.DMNocAsyncRead,
    data_movement.DMNocAsyncWrite,
    data_movement.DMNocAsyncReadBarrier,
    data_movement.DMNocAsyncWriteBarrier,
    circular_buffer.CBWaitFront,
    circular_buffer.CBPopFront,
    circular_buffer.CBPushBack,
    circular_buffer.CBReserveBack,
    *compute.Compute.operations,
]


# expressions return a value
# TODO: Greyed out ones don't work as expected due to new printing technique
TenstorrentExpr = [
    ttshared.GetCompileTimeArgVal,
    # data_movement.DMGetNocAddrFromBankId,
    circular_buffer.CBPagesAvailableAtFront,
    circular_buffer.CBPagesReservableAtBack,
    # circular_buffer.CBGetWritePointer,
    # circular_buffer.CBGetReadPointer,
    host.TTHostCore,
    # host.TTCreateDevice,
    # host.TTCreateCBConfig,
    # host.TTCreateCircularBuffer,
    host.TTGetCommandQueue,
    # host.TTCreateProgram,
    host.TTCreateDRAMConfig,
    # host.TTCreateBuffer,
    # host.TTCreateKernel,
    # host.TTCreateComputeKernel,
    host.TTGetMemoryAddress,
]

CMP_PREDICATE_TO_SYMBOL = ["==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">="]

ARITH_OP_TO_SYM = {
    arith.AddiOp: "+",
    arith.MuliOp: "*",
    arith.AddfOp: "+",
    arith.MulfOp: "*",
    arith.AndIOp: "&&",
    arith.OrIOp: "||",
    arith.XOrIOp: "^",
    arith.SubiOp: "-",
    arith.SubfOp: "-",
    arith.DivfOp: "/",
}

Expressions = [
    arith.ConstantOp,
    arith.AddiOp,
    arith.MuliOp,
    arith.AddfOp,
    arith.MulfOp,
    arith.IndexCastOp,
    arith.CmpiOp,
    arith.AndIOp,
    arith.OrIOp,
    arith.XOrIOp,
    arith.SubiOp,
    arith.SubfOp,
    arith.SIToFPOp,
    arith.ExtUIOp,
    arith.DivfOp,
    memref.LoadOp,
    scf.YieldOp,
    *TenstorrentExpr,
]

Expressions = list(map(lambda x: x.__name__, Expressions))

uint32 = builtin.IntegerType(32, signedness=builtin.Signedness.UNSIGNED)
uint64 = builtin.IntegerType(64, signedness=builtin.Signedness.UNSIGNED)

MLIR_TO_CPP_TYPES = {
    builtin.IndexType(): "std::int32_t",
    builtin.i32: "std::int32_t",
    uint32: "uint32_t",
    builtin.i64: "std::int64_t",
    uint64: "uint64_t",
    builtin.f32: "float",
    builtin.i1: "bool",
    host.CoreCoord(): "CoreCoord",
    host.Device(): "IDevice*",
    host.CommandQueue(): "CommandQueue &",
    host.Program(): "Program",
    host.Buffer(): "std::shared_ptr<Buffer>",
    host.Kernel(): "KernelHandle",
    host.CircularBufferConfig(): "CircularBufferConfig",
    host.CBHandle(): "CBHandle",
}

TYPE_STR_TO_TT_DATA_FORMAT = {"int": "Int32"}


def get_api_name(op_name: str) -> str:
    dialect_prefix = op_name.split(".")[0]
    match dialect_prefix:
        case "cb":
            return op_name.replace(".", "_")
        case "dm":
            return op_name.replace("dm.", "")
        case "comp":
            return op_name.replace("comp.", "")
        case "ttshared":
            return op_name.replace("ttshared.", "")
        case default:
            raise Exception(f"Unhandled operation name: {op_name}")


class PrintMetalium:
    """
    Prints the Tenstorrent Metalium API (C) given a list of xDSL operations
    """

    def __init__(self, write_files: bool = False):
        self._indent = 0
        self._writing_files = write_files
        self.date = dt.now().strftime("%d-%m-%y--%H-%M")
        self._file = None
        self._names = {}  # SSAVal -> Variable Name
        self._free_end_of_fn = []
        self._unique_name_ctr = 0

        self._kernel_type = []
        self._skip_next_op = False

    def is_host(self):
        return self._kernel_type[-1] == "host"

    def is_compute(self):
        return self._kernel_type[-1] == "compute"

    def is_data_in(self):
        return self._kernel_type[-1] == "data_in"

    def is_data_out(self):
        return self._kernel_type[-1] == "data_out"

    def is_device(self):
        return self.is_compute() or self.is_data_out() or self.is_data_in()

    def is_unknown(self):
        return self._kernel_type[-1] == "unknown"

    def print_include(self, lib: str):
        self.print(f"#include {lib}", end="\n")

    def print_op(self, operation):
        if self._skip_next_op:
            self._skip_next_op = False
            return

        if type(operation).__name__ in Expressions:
            return

        if isa(operation, builtin.ModuleOp):
            if "kernel_type" in operation.attributes:
                kernel_type = operation.attributes["kernel_type"].data
                self._kernel_type.append(kernel_type)

                if self._writing_files:
                    os.makedirs(self.date, exist_ok=True)
                    self._file = open(f"{self.date}/{kernel_type}.cpp", "w")

                if kernel_type == "host":
                    self.print_include('"host_api.hpp"')
                    self.print_include('"device_impl.hpp"')
                    self.println("\nusing namespace tt;")
                    self.println("using namespace tt::tt_metal;\n")

                elif kernel_type == "data_in" or kernel_type == "data_out":
                    self.print_include("<stdint.h>")
                    self.print_include('"dataflow_api.h"')
                    self.print_include('"debug/dprint.h"')

                # TODO: generalise based on code possible? MLIR ops for include? Pass that adds these?
                elif kernel_type == "compute":
                    self.print_include("<cstdint>")
                    self.print_include('"compute_kernel_api/matmul.h"')
                    self.print_include('"compute_kernel_api/tile_move_copy.h"')
                    self.print_include('"compute_kernel_api/eltwise_binary.h"')
                    self.print_include('"debug/dprint.h"')
                    self.println("\nnamespace NAMESPACE {")
            else:
                self._kernel_type.append("unknown")
            for region in operation.regions:
                for block in region.blocks:
                    self.print_op(block)

            # no longer compiling that same kernel
            if self.is_unknown() and self._writing_files:
                self._file.close()
                self._file = None

            if self.is_compute():
                self.println("\n}")  # closes namespace

            self._kernel_type.pop()
        elif isa(operation, Block):
            for op in operation.ops:
                self.print_op(op)
        elif isa(operation, func.FuncOp):
            self.print_func_def(operation)
        elif isa(operation, printf.PrintFormatOp):
            self.print_print(operation)
        elif isa(operation, func.ReturnOp):
            self.print_return(operation)
        elif isa(operation, memref.AllocOp) or isa(operation, memref.AllocaOp):
            self.print_declaration(operation)
        elif isa(operation, memref.StoreOp):
            self.print_assignment(operation)
        elif isa(operation, scf.ForOp):
            self.print_for_loop(operation)
        elif isa(operation, scf.IfOp):
            self.print_if_statement(operation)
        elif isa(operation, host.TTEnqueueWriteBuffer) or isa(
            operation, host.TTEnqueueReadBuffer
        ):
            self.print_ttenqueue_readwrite_buffer(operation)
        elif isa(operation, host.TTEnqueueProgram):
            self.print_ttenqueue_program(operation)
        elif isa(operation, host.TTFinish):
            self.print_ttfinish(operation)
        elif isa(operation, host.TTCloseDevice):
            self.print_ttclose_device(operation)
        elif isa(operation, host.TTSetRuntimeArgs):
            self.print_ttset_runtime_args(operation)
        elif isa(operation, host.TTCreateProgram):
            self.print_ttcreate_program(operation)
        elif isa(operation, host.TTCreateDevice):
            self.print_ttcreate_device(operation)
        elif isa(operation, host.TTCreateCBConfig):
            self.print_ttcreate_cb_config(operation)
        elif isa(operation, host.TTCreateBuffer):
            self.print_ttcreate_buffer(operation)
        elif isa(operation, host.TTCreateCircularBuffer):
            self.print_ttcreate_circular_buffer(operation)
        elif isa(operation, host.TTCreateKernel) or isa(
            operation, host.TTCreateComputeKernel
        ):
            self.print_ttcreate_kernel(operation)
        elif isa(operation, circular_buffer.CBGetWritePointer):
            self.print_cb_get_write_pointer(operation)
        elif isa(operation, circular_buffer.CBGetReadPointer):
            self.print_cb_get_read_pointer(operation)
        elif isa(operation, data_movement.DMGetNocAddrFromBankId):
            self.print_ttget_noc_addr_from_bank_id(operation)
        elif isa(operation, builtin.UnrealizedConversionCastOp):
            self.print_unrealized_conversion_cast(operation)
        elif type(operation) in TenstorrentStmts:
            self.print_tt_stmt_generic(operation)

        else:
            raise NotImplementedError(
                f"Unhandled operation: {operation.__class__.__name__}"
            )

    def print_return(self, op):
        if len(op.arguments) > 0:
            assert len(op.arguments) == 1
            self.print("return ", True)
            self.print_expr(op.arguments[0])
            self.print(";", end="\n")

    def print_ttset_runtime_args(self, op):
        self.print("SetRuntimeArgs(", True)
        self.print_expr(op.program)
        self.print(", ")
        self.print_expr(op.kernel)
        self.print(", ")
        self.print_expr(op.core)
        self.print(", {")
        for idx, arg in enumerate(op.args):
            if idx > 0:
                self.print(", ")
            self.print_expr(arg)
        self.print("});", end="\n")

    def print_ttclose_device(self, op):
        self.print("CloseDevice(", True)
        self.print_expr(op.device)
        self.print(");", end="\n")

    def print_ttfinish(self, op):
        self.print("Finish(", True)
        self.print_expr(op.command_queue)
        self.print(");", end="\n")

    def print_ttenqueue_program(self, op):
        self.print("EnqueueProgram(", True)
        self.print_expr(op.command_queue)
        self.print(", ")
        self.print_expr(op.program)
        self.print(", ")
        self.print_expr(op.blocking)
        self.print(");", end="\n")

    def print_ttenqueue_readwrite_buffer(self, op):
        if isa(op, host.TTEnqueueWriteBuffer):
            self.print("EnqueueWriteBuffer(", True)
        elif isa(op, host.TTEnqueueReadBuffer):
            self.print("EnqueueReadBuffer(", True)
        else:
            assert False
        self.print_expr(op.command_queue)
        self.print(", ")
        self.print_expr(op.buffer)
        self.print(", ")
        self.print_expr(op.data)
        self.print(", ")
        self.print_expr(op.blocking)
        self.print(");", end="\n")

    def print_expr(self, ssa_val):
        expr = ssa_val.owner
        if isa(expr, arith.ConstantOp):
            if (
                isa(expr.result.type, builtin.IntegerType)
                and expr.result.type.width.data == 1
            ):
                self.print(f"{'false' if expr.value.value.data == 0 else 'true'}")
            else:
                self.print(str(expr.value.value.data))
        elif isa(expr, host.TTHostCore):
            self.print_tthost_core(expr)
        elif isa(expr, host.TTCreateDevice):
            self.print_ttcreate_device(expr)
        elif isa(expr, host.TTGetCommandQueue):
            self.print_ttget_command_queue(expr)
        elif isa(expr, host.TTCreateProgram):
            self.print_ttcreate_program(expr)
        elif isa(expr, host.TTCreateDRAMConfig):
            self.print_ttcreate_dram_config(expr)
        elif isa(expr, host.TTCreateCBConfig):
            self.print_ttcreate_cb_config(expr)
        elif isa(expr, host.TTCreateBuffer):
            self.print_ttcreate_buffer(expr)
        elif isa(expr, host.TTCreateCircularBuffer):
            self.print_ttcreate_circular_buffer(expr)
        elif isa(expr, host.TTCreateKernel) or isa(expr, host.TTCreateComputeKernel):
            self.print_ttcreate_kernel(expr)
        elif isa(expr, host.TTGetMemoryAddress):
            self.print_ttget_memory_address(expr)
        elif isa(expr, data_movement.DMGetNocAddrFromBankId):
            self.print_ttget_noc_addr_from_bank_id(expr)
        elif isa(expr, circular_buffer.CBGetWritePointer):
            self.print_cb_get_write_pointer(expr)
        elif isa(expr, circular_buffer.CBGetReadPointer):
            self.print_cb_get_read_pointer(expr)
        elif isa(expr, memref.LoadOp):
            self.print_load_variable(expr)
        elif isa(expr, Block):
            self.print(f"fn_arg_{ssa_val.index}")
        elif isa(expr, memref.AllocaOp) or isa(expr, memref.AllocOp):
            self.print(expr.results[0].name_hint)
        elif isa(expr, BinaryOperation):
            self.print_binary_op(expr)
        elif isa(expr, arith.SIToFPOp):
            self.print_cast_to_float(expr)
        elif isa(expr, arith.ExtUIOp):
            self.print_cast_integer(expr)
        elif isa(expr, builtin.UnrealizedConversionCastOp):
            self.print_unrealized_conversion_cast(expr, is_expr=True)
        elif isa(expr, arith.IndexCastOp):
            # Go directly to the operation used as an input and process this
            self.print_expr(expr.input)
        elif type(expr) in TenstorrentExpr:
            self.print_tt_expr_generic(expr)
        else:
            raise NotImplementedError(
                f"Unhandled expression: {expr.__class__.__name__}"
            )

    def print_unrealized_conversion_cast_expr(self, op):
        operand = op.inputs[0]
        in_type = operand.type
        out_type = op.outputs[0].type

        constexpr = "ttshared.constexpr"
        if in_type.name == constexpr or out_type.name == constexpr:
            self.print_expr(operand)
            return

        in_int = in_type.name == "integer_type"
        out_int = out_type.name == "integer_type"

        if in_int:
            in_sign = in_type.signedness.data

        if out_int:
            out_sign = out_type.signedness.data

        width = in_type.width.data

        if in_int and not out_int:
            # casting from i32, si32, ui32 to float
            assert width == 32
            self.print_cast_to_float(op)
            return

        if in_int and in_sign == Signedness.UNSIGNED:
            # we know i32, si32 become int32_t so we need to cast
            # uint32 -> int32
            self.print(f"static_cast<std::int{width}_t>(")
            self.print_expr(operand)
            self.print(")")
            return

        if in_int:
            # here the int is signless/signed => int32
            # also out_int == True
            if out_sign == Signedness.UNSIGNED:
                self.print(f"static_cast<std::uint{width}_t>(")
                self.print_expr(operand)
                self.print(")")

            return

        if not in_int:
            if out_sign == Signedness.UNSIGNED:
                self.print(f"static_cast<std::uint{width}_t>(")
                self.print_expr(operand)
                self.print(")")
            else:
                self.print(f"static_cast<std::int{width}_t>(")
                self.print_expr(operand)
                self.print(")")

    def print_unrealized_conversion_cast_stmt(self, op):
        ssa = op.results[0]
        if isa(ssa.type, builtin.MemRefType):
            assert ssa.type.element_type in MLIR_TO_CPP_TYPES
            type_str = MLIR_TO_CPP_TYPES[ssa.type.element_type]
            var_name = self.get_or_create_varname(ssa)

            self.print(f"{type_str} * {var_name} = ({type_str}*) ", True)
            self.print_expr(op.inputs[0])
            self.print(";", end="\n")

    def print_unrealized_conversion_cast(self, op, is_expr=False):
        if is_expr:
            self.print_unrealized_conversion_cast_expr(op)
            return

        self.print_unrealized_conversion_cast_stmt(op)

    def print_print(self, op):
        string = op.format_str.data

        # compiling a separate function, hard to tell, could make assumptions
        # based on other functions in the file, etc. For now crash
        if self.is_unknown():
            raise ValueError("Can only print in functions marked with @tt.decorator")
        elif self.is_host():
            self.print(f'printf("{string}\\n");', True, end="\n")
        elif self.is_data_in():
            self.print(
                f'DPRINT_DATA0(DPRINT << "{string}" << ENDL());',
                True,
                end="\n",
            )
        elif self.is_data_out():
            self.print(
                f'DPRINT_DATA1(DPRINT << "{string}" << ENDL());',
                True,
                end="\n",
            )
        elif self.is_compute():
            self.print(f'DPRINT_MATH(DPRINT << "{string}" << ENDL());', True, end="\n")
        else:
            raise ValueError(f"Unknown kernel type: {self._kernel_type[-1]}")

    def print_cb_get_write_pointer(self, op):
        if op.results[0] in self._names.keys():
            self.print(self._names[op.results[0]])
        else:
            tgt_name = op.results[0].name_hint
            if tgt_name is None or tgt_name in self._names.values():
                tgt_name = str(self._unique_name_ctr)
                self._unique_name_ctr += 1
            if tgt_name.isdigit():
                tgt_name = "write_ptr_" + tgt_name
            assert op.results[0].type in MLIR_TO_CPP_TYPES
            self.print(f"{MLIR_TO_CPP_TYPES[op.results[0].type]} {tgt_name} = ", True)
            self.print("get_write_ptr(")
            self.print_expr(op.cb_id)
            self.print(");", end="\n")
            self._names[op.results[0]] = tgt_name

    def print_cb_get_read_pointer(self, op):
        if op.results[0] in self._names.keys():
            self.print(self._names[op.results[0]])
        else:
            tgt_name = op.results[0].name_hint
            if tgt_name is None or tgt_name in self._names.values():
                tgt_name = str(self._unique_name_ctr)
                self._unique_name_ctr += 1
            if tgt_name.isdigit():
                tgt_name = "read_ptr_" + tgt_name
            assert op.results[0].type in MLIR_TO_CPP_TYPES
            self.print(f"{MLIR_TO_CPP_TYPES[op.results[0].type]} {tgt_name} = ", True)
            self.print("get_read_ptr(")
            self.print_expr(op.cb_id)
            self.print(");", end="\n")
            self._names[op.results[0]] = tgt_name

    def print_load_variable(self, op):
        self.print(op.memref.name_hint)  # TODO: should this be varname lookup?
        if len(op.indices) > 0:
            # For now we limit ourselves to one dimensional arrays
            assert len(op.indices) == 1
            self.print("[")
            self.print_expr(op.indices[0])
            self.print("]")

    def print_cast_to_float(self, op):
        self.print("static_cast<float>(")
        self.print_expr(op.operands[0])
        self.print(")")

    def print_cast_integer(self, op):
        self.print(f"static_cast<{MLIR_TO_CPP_TYPES[op.results[0].type]}>(")
        self.print_expr(op.operands[0])
        self.print(")")

    def print_tthost_core(self, op):
        self.print("CoreCoord{")
        self.print_expr(op.src_noc_x)
        self.print(", ")
        self.print_expr(op.src_noc_y)
        self.print("}")

    def print_ttget_memory_address(self, op):
        self.print_expr(op.buffer)
        self.print("->address()")

    def print_ttget_noc_addr_from_bank_id(self, op):
        if op.results[0] in self._names.keys():
            self.print(self._names[op.results[0]])
        else:
            tgt_name = op.results[0].name_hint
            if tgt_name is None or tgt_name in self._names.values():
                tgt_name = str(self._unique_name_ctr)
                self._unique_name_ctr += 1
            if tgt_name.isdigit():
                tgt_name = "noc_addr_" + tgt_name
            assert op.results[0].type in MLIR_TO_CPP_TYPES
            self.print(f"{MLIR_TO_CPP_TYPES[op.results[0].type]} {tgt_name} = ", True)
            self.print_tt_operation_generic(op, False)
            self.print(";", end="\n")
            self._names[op.results[0]] = tgt_name

    def print_ttcreate_kernel(self, op):
        if op.results[0] in self._names.keys():
            self.print(self._names[op.results[0]])
        else:
            tgt_name = op.results[0].name_hint
            if tgt_name is None or tgt_name in self._names.values():
                tgt_name = str(self._unique_name_ctr)
                self._unique_name_ctr += 1
            if tgt_name.isdigit():
                tgt_name = "kernel_" + tgt_name
            assert op.results[0].type in MLIR_TO_CPP_TYPES
            self.print(
                f"{MLIR_TO_CPP_TYPES[op.results[0].type]} {tgt_name} = CreateKernel(",
                True,
            )
            self.print_expr(op.program)
            self.print(f", {op.kernel_name}, ")
            self.print_expr(op.core)
            self.print(", ")

            rv_core_flag = list(op.riscv_core.flags)[0]
            if (
                rv_core_flag == host.RISCVCoreFlags.DATAMOVEMENT_0
                or rv_core_flag == host.RISCVCoreFlags.DATAMOVEMENT_1
            ):
                self.print(
                    "DataMovementConfig{.processor = DataMovementProcessor::RISCV_"
                )
                if rv_core_flag == host.RISCVCoreFlags.DATAMOVEMENT_0:
                    self.print("0")
                else:
                    self.print("1")

                self.print(f", .noc=NOC::RISCV_{op.noc_id.data}_default}}")

            if rv_core_flag == host.RISCVCoreFlags.COMPUTE:
                a = "ComputeConfig {"
                b = f".math_fidelity = MathFidelity::{op.math_fidelity.data[0].value}, "
                c = f".fp32_dest_acc_en = {str(bool(op.fp32_dest_acc_en.value.data)).lower()}, "
                d = f".math_approx_mode = {str(bool(op.math_approx_mode.value.data)).lower()}, "
                e = ".compile_args = {}"
                f = "}"
                for s in [a, b, c, d, e, f]:
                    self.print(s)

            self.print(");", end="\n")
            self._names[op.results[0]] = tgt_name

    def print_ttcreate_circular_buffer(self, op):
        if op.results[0] in self._names.keys():
            self.print(self._names[op.results[0]])
        else:
            tgt_name = op.results[0].name_hint
            if tgt_name is None or tgt_name in self._names.values():
                tgt_name = str(self._unique_name_ctr)
                self._unique_name_ctr += 1
            if tgt_name.isdigit():
                tgt_name = "cb_" + tgt_name
            assert op.results[0].type in MLIR_TO_CPP_TYPES
            self.print(
                f"{MLIR_TO_CPP_TYPES[op.results[0].type]} {tgt_name} = tt_metal::CreateCircularBuffer(",
                True,
            )
            self.print_expr(op.program)
            self.print(", ")
            self.print_expr(op.core)
            self.print(", ")
            self.print_expr(op.config)
            self.print(");", end="\n")
            self._names[op.results[0]] = tgt_name

    def print_ttcreate_buffer(self, op):
        if op.results[0] in self._names.keys():
            self.print(self._names[op.results[0]])
        else:
            tgt_name = op.results[0].name_hint
            if tgt_name is None or tgt_name in self._names.values():
                tgt_name = str(self._unique_name_ctr)
                self._unique_name_ctr += 1
            if tgt_name.isdigit():
                tgt_name = "buffer_" + tgt_name
            assert op.results[0].type in MLIR_TO_CPP_TYPES
            self.print(
                f"{MLIR_TO_CPP_TYPES[op.results[0].type]} {tgt_name} = CreateBuffer(",
                True,
            )
            self.print_expr(op.config)
            self.print(");", end="\n")
            self._names[op.results[0]] = tgt_name

    def print_ttcreate_dram_config(self, op):
        self.print("{")
        self.print(".device=device")  # TODO: this doesn't work, need to fix
        self.print(", .size=")
        self.print_expr(op.size)
        self.print(", .page_size=")
        self.print_expr(op.page_size)
        self.print(", .buffer_type = BufferType::DRAM")
        self.print("}")

    def print_ttcreate_cb_config(self, op):
        if op.results[0] in self._names.keys():
            self.print(self._names[op.results[0]])
        else:
            tgt_name = op.results[0].name_hint
            if tgt_name is None or tgt_name in self._names.values():
                tgt_name = str(self._unique_name_ctr)
                self._unique_name_ctr += 1
            if tgt_name.isdigit():
                tgt_name = "cb_config_" + tgt_name

            assert op.results[0].type in MLIR_TO_CPP_TYPES
            self.print(
                f"{MLIR_TO_CPP_TYPES[op.results[0].type]} {tgt_name} = CircularBufferConfig(",
                True,
            )
            self.print_expr(op.num_buffers)
            self.print("*")
            self.print_expr(op.page_size)
            self.print(", {{")
            self.print_expr(op.cb_index)
            assert op.data_type.data in TYPE_STR_TO_TT_DATA_FORMAT
            self.print(
                ", tt::DataFormat::"
                + TYPE_STR_TO_TT_DATA_FORMAT[op.data_type.data]
                + "}}).set_page_size("
            )
            self.print_expr(op.cb_index)
            self.print(", ")
            self.print_expr(op.page_size)
            self.print(");", end="\n")
            self._names[op.results[0]] = tgt_name

    def print_ttcreate_device(self, op):
        if op.results[0] in self._names.keys():
            self.print(self._names[op.results[0]])
        else:
            tgt_name = op.results[0].name_hint
            if tgt_name is None or tgt_name in self._names.values():
                tgt_name = str(self._unique_name_ctr)
                self._unique_name_ctr += 1
            if tgt_name.isdigit():
                tgt_name = "device_" + tgt_name
            assert op.results[0].type in MLIR_TO_CPP_TYPES
            self.print(
                f"{MLIR_TO_CPP_TYPES[op.results[0].type]} {tgt_name} = CreateDevice(",
                True,
            )
            self.print_expr(op.index)
            self.print(");", end="\n")
            self._names[op.results[0]] = tgt_name

    def print_ttcreate_program(self, op):
        if op.results[0] in self._names.keys():
            self.print(self._names[op.results[0]])
        else:
            tgt_name = op.results[0].name_hint
            if tgt_name is None or tgt_name in self._names.values():
                tgt_name = str(self._unique_name_ctr)
                self._unique_name_ctr += 1
            if tgt_name.isdigit():
                tgt_name = "program_" + tgt_name
            assert op.results[0].type in MLIR_TO_CPP_TYPES
            self.print(
                f"{MLIR_TO_CPP_TYPES[op.results[0].type]} {tgt_name} = CreateProgram();",
                True,
                end="\n",
            )
            self._names[op.results[0]] = tgt_name

    def print_ttget_command_queue(self, op):
        assert isa(op.device.owner, memref.LoadOp) or isa(
            op.device.owner, host.TTCreateDevice
        )
        if isa(op.device.owner, memref.LoadOp):
            self.print(f"{op.device.owner.memref.name_hint}->command_queue()")
        elif isa(op.device.owner, host.TTCreateDevice):
            self.print_expr(op.device)
            self.print("->command_queue()")

    def print_binary_op(self, op):
        if isinstance(op, arith.ComparisonOperation):
            self.print_expr(op.lhs)

            assert op.predicate.value.data < len(CMP_PREDICATE_TO_SYMBOL)
            print(f" {CMP_PREDICATE_TO_SYMBOL[op.predicate.value.data]} ", end="")

            self.print_expr(op.rhs)
        else:
            if isa(op, arith.XOrIOp):
                self.print("!")

            self.print_expr(op.lhs)

            if not isa(op, arith.XOrIOp):
                self.print(f" {ARITH_OP_TO_SYM[op.__class__]} ", end="")

                self.print_expr(op.rhs)

    def print_func_def(self, func_op: func.FuncOp):
        """
        void func_name(typea a, typeb b, ...) {

        }
        """
        return_type = "void"

        if len(func_op.function_type.outputs) > 0:
            assert len(func_op.function_type.outputs) == 1
            return_type = MLIR_TO_CPP_TYPES[func_op.function_type.outputs.data[0]]

        vis = func_op.parent_op().attributes.get("vis")
        external_host = self.is_host() and vis and vis.data == "external"
        if external_host:
            return_type = 'extern "C" ' + return_type

        func_name = func_op.sym_name.data
        if self.is_data_in() or self.is_data_out():
            func_name = "kernel_main"
        if self.is_compute():
            func_name = "MAIN"
        if external_host:
            func_name = "host_entry"
        if self.is_host() and not external_host:
            func_name = "main"

        self.print(f"\n{return_type} {func_name}", True)

        if not self.is_compute():
            self.print("(")

        if self.is_host():
            for idx, input_type in enumerate(func_op.function_type.inputs):
                is_ref = isa(input_type, builtin.MemRefType)
                if is_ref:
                    input_type = input_type.element_type
                    if isa(input_type, builtin.MemRefType):
                        input_type = input_type.element_type

                type_decl = MLIR_TO_CPP_TYPES[input_type]
                if idx > 0:
                    self.print(", ")
                self.print(f"{type_decl}{'*' if is_ref else ''} fn_arg_{idx}")

        if not self.is_compute():
            self.print(")")

        self.print(" {", end="\n")
        self.indent()

        if self.is_device():
            for idx, input in enumerate(func_op.function_type.inputs):
                type_decl = MLIR_TO_CPP_TYPES[input]
                self.print(
                    f"{type_decl} fn_arg_{idx} = get_arg_val<{type_decl}>({idx});",
                    True,
                    end="\n",
                )

        # Have to do some messing around here with the return operation, which must be the last operation. That is
        # because we insert free for heap allocated memory, but that must be before the return, hence extract this,
        # print the rest, then print the free and lastly print the return
        ret_op = func_op.body.block.ops.last
        assert isa(ret_op, func.ReturnOp)

        self.print_region(list(func_op.body.block.ops)[:-1])

        for free_c in self._free_end_of_fn:
            self.print(f"free({free_c});", True, end="\n")

        self.print_op(ret_op)

        self.dedent()

        self._free_end_of_fn = []

        self.print("}", True, end="\n")

    def print_for_loop(self, loop: scf.ForOp):
        # we know the first operation in the loop should be the store into i
        store_i = loop.body.block.first_op
        loop_index_name = store_i.operands[1].name_hint

        self.print(f"for ({loop_index_name} = ", True)
        self.print_expr(loop.lb)
        self.print(f"; {loop_index_name} < ")
        self.print_expr(loop.ub)
        self.print(f"; {loop_index_name} += ")
        self.print_expr(loop.step)
        self.print(") {", end="\n")

        self.indent()
        # Remove the first operation as this assigns the loop variable and
        # we have handled this above already by looking into that operation and
        # extracting the name
        self.print_region(list(loop.body.ops)[1:])
        self.dedent()

        self.print("}", True, end="\n")

    def print_region(self, body: Region | list):
        if isa(body, list):
            for op in body:
                self.print_op(op)
        else:
            for block in body.blocks:
                self.print_op(block)

    def print_declaration(self, op: memref.AllocOp | memref.AllocaOp):
        if isa(op.result_types[0].element_type, host.DRAMBufferConfig):
            # Need to handle this differently as it is not allocated and this is
            # done in the assignment
            return

        var_name = self.get_or_create_varname(op.results[0])
        mlir_type = op.result_types[0].element_type

        modifier = ""

        if isinstance(mlir_type, ttshared.ConstExprType):
            modifier = "constexpr "
            mlir_type = mlir_type.get_element_type()

        type_decl = MLIR_TO_CPP_TYPES[mlir_type]
        scalar = len(op.result_types[0].shape) == 0

        if type_decl == "Program":
            type_decl += "&"

        if scalar:
            self.print(modifier + type_decl + " " + var_name, True)

            if PrintMetalium.is_decl_init(op):
                self.print(" = ")
                self.print_expr(op.next_op.operands[0])
                op.next_op.attributes["ignore"] = True

            self.print(";", end="\n")

        else:
            total_size = 1
            for s in op.result_types[0].shape:
                # Might not be correct for multi-dimensional arrays
                total_size *= s.data
            if isa(op, memref.AllocOp):
                self.print(
                    f"{type_decl} * {var_name} = ({type_decl}*) malloc(sizeof({type_decl})*{total_size});",
                    True,
                    end="\n",
                )
                self._free_end_of_fn.append(var_name)
            elif isa(op, memref.AllocaOp):
                self.print(
                    f"{type_decl} {var_name}[{total_size}];",
                    True,
                    end="\n",
                )
            else:
                assert False

    @staticmethod
    def is_decl_init(operation: Operation):
        is_decl = isa(operation, memref.AllocOp | memref.AllocaOp)
        is_init = isa(operation.next_op, memref.StoreOp)

        if is_decl and is_init:
            return operation.next_op.operands[1] == operation.results[0]

        return False

    def print_assignment(self, op: memref.StoreOp):
        if "ignore" in op.attributes:
            if op.attributes["ignore"]:
                return
        ssa_value = op.operands[0]
        ssa_destination = op.operands[1]
        varname = self.get_or_create_varname(ssa_destination)

        if isa(ssa_destination.type.element_type, host.DRAMBufferConfig):
            self.print(f"InterleavedBufferConfig {varname} ", True)

        else:
            self.print(f"{varname}", True)
            if len(op.indices) > 0:
                # For now we limit ourselves to one dimensional arrays
                assert len(op.indices) == 1
                self.print("[")
                self.print_expr(op.indices[0])
                self.print("]")
            self.print(" = ")

        self.print_expr(ssa_value)
        self.print(";", end="\n")

    def print_if_statement(self, op: scf.IfOp):
        self.print("if (", True)
        self.print_expr(op.cond)
        self.print(") {", end="\n")

        self.indent()
        self.print_op(op.true_region.blocks[0])
        self.dedent()

        or_else = len(op.false_region.blocks) > 0

        self.print("}" + (" else {" if or_else else ""), True, end="\n")

        if or_else:
            self.indent()
            self.print_region(op.false_region)
            self.dedent()
            self.print("}", True, end="\n")

    def _create_varname(self, ssa: SSAValue, hint: str = None) -> str:
        names = self._names.values()
        assert ssa not in self._names

        if not hint:
            hint = ssa.name_hint if ssa.name_hint else "a"

        count = 0
        name = hint
        while name in names:
            count += 1
            name = hint + str(count)

        self._names[ssa] = name
        return name

    def _get_varname(self, ssa: SSAValue) -> str | None:
        if ssa in self._names:
            return self._names[ssa]

        return None

    def get_or_create_varname(self, ssa: SSAValue, hint: str = None) -> str:
        """
        Provided name hint overrides the ssa.name_hint.
        """
        if ssa not in self._names:
            return self._create_varname(ssa, hint)

        return self._names[ssa]

    def print_tt_expr_generic(self, expression):
        self.print_tt_operation_generic(expression, False)

    def print_tt_stmt_generic(self, operation):
        self.print_tt_operation_generic(operation, True)

    def print_tt_operation_generic(self, operation, is_statement):
        api_name = get_api_name(operation.name)
        self.print(api_name, is_statement)
        if operation.properties:
            self.print("<")
            for idx, p in enumerate(operation.properties.values()):
                if idx > 0:
                    self.print(", ")
                self.print(str(p))
            self.print(">")
        self.print("(")
        for idx, expr in enumerate(operation.operands):
            if idx > 0:
                self.print(", ")
            self.print_expr(expr)
        self.print(")")
        if is_statement:
            self.print(";", end="\n")

    def print(self, s: str, indented: bool = False, end=""):
        prefix = self._prefix if indented else ""
        if self._writing_files and self._file:
            print(prefix + s, file=self._file, end=end)
        else:
            print(prefix + s, end=end)

    def println(self, s: str, indented: bool = False):
        self.print(s, indented, end="\n")

    def indent(self):
        self._indent += 1

    def dedent(self):
        self._indent -= 1

    @property
    def _prefix(self):
        return " " * 4 * self._indent

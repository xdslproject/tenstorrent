from xdsl.ir import Block, Region, OpResult, Attribute, SSAValue, BlockArgument
from xdsl.irdl import IRDLOperation

from xdsl.utils.hints import isa
from tenstorrent.utils import flatten

import xdsl.dialects.arith as arith
import xdsl.dialects.memref as memref
import xdsl.dialects.scf as scf
import xdsl.dialects.func as func
import xdsl.dialects.builtin as builtin

import tenstorrent.dialects.host as host
import tenstorrent.dialects.circular_buffer as circular_buffer
import tenstorrent.dialects.data_movement as data_movement
import tenstorrent.dialects.compute as compute


ArithmeticOperation = arith.SignlessIntegerBinaryOperation | arith.FloatingPointLikeBinaryOperation
BooleanOperation = arith.AndIOp | arith.OrIOp | arith.CmpiOp | arith.CmpfOp
BinaryOperation = ArithmeticOperation | BooleanOperation
OpWithBody = func.FuncOp | scf.ForOp | scf.WhileOp
CircularBufferOperationWithResult = circular_buffer.CBPagesAvailableAtFront | circular_buffer.CBPagesReservableAtBack

TRUE = builtin.IntegerAttr.from_int_and_width(1, 1)

TenstorrentOps = [data_movement.DMNocAsyncRead, data_movement.DMNocAsyncWrite, data_movement.DMNocAsyncReadBarrier, data_movement.DMNocAsyncWriteBarrier]

TenstorrentExpr = [data_movement.DMGetNocAddrFromBankId]

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

SkipOps = [
            arith.ConstantOp, memref.LoadOp, arith.AddiOp, arith.MuliOp, arith.AddfOp, arith.MulfOp, arith.IndexCastOp, scf.YieldOp,
            arith.CmpiOp, arith.AndIOp, arith.OrIOp, arith.XOrIOp, arith.SubiOp, arith.SubfOp, arith.ExtFOp, arith.DivfOp,
            circular_buffer.CBPagesReservableAtBack, circular_buffer.CBPagesAvailableAtFront, host.TTHostCore, host.TTCreateDevice,
            host.TTGetCommandQueue, host.TTCreateProgram, host.TTCreateDRAMConfig, host.TTCreateBuffer, host.TTCreateKernel, host.TTGetMemoryAddress,
            *TenstorrentExpr
        ]

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
            host.Device(): "Device*",
            host.CommandQueue(): "CommandQueue &",
            host.Program(): "Program",
            host.Buffer(): "std::shared_ptr<Buffer>",
            host.Kernel(): "KernelHandle",
        }


def get_api_name(op_name: str) -> str:
    first_two_chars = op_name[:2]
    match first_two_chars:
        case 'cb':
            return op_name.replace('.', '_')
        case 'dm':
            return op_name.replace('dm.', '')
        case 'co':
            return op_name.replace('comp.', '')
        case default:
            raise Exception(f"Unhandled operation name: {op_name}")


class PrintMetalium:
    """
    Prints the Tenstorrent Metalium API (C) given a list of xDSL operations
    """
    def __init__(self, file=None):
        self._indent = 0
        self._file = file
        self._names = {}  # SSAVal -> Variable Name
        self._free_end_of_fn=[]

        self._skip_next_op = False

    def print_op(self, operation):
        if self._skip_next_op:
            self._skip_next_op = False
            return

        if type(operation) in SkipOps:
            return

        if isa(operation, builtin.ModuleOp):
            if "kernel_type" in operation.attributes:
              if operation.attributes["kernel_type"].data == "host":
                self.print("#include \"tt_metal/host_api.hpp\"", indented=True, end='\n')
                self.print("#include \"tt_metal/impl/device/device.hpp\"", indented=True, end='\n')
                self.print("#include \"tt_metal/common/bfloat16.hpp\"", indented=True, end='\n')
                self.print("\nusing namespace tt;", indented=True, end='\n')
                self.print("using namespace tt::tt_metal;\n", indented=True, end='\n')
              elif operation.attributes["kernel_type"].data == "data_in":
                self.print("#include <stdint.h>", indented=True, end='\n')
                self.print("#include \"dataflow_api.h\"\n", indented=True, end='\n')
            for region in operation.regions:
                for block in region.blocks:
                    self.print_op(block)
        elif isa(operation, Block):
            for op in operation.ops:
                self.print_op(op)
        elif isa(operation, func.FuncOp):
            self.print_func_def(operation)
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
        elif isa(operation, host.TTEnqueueWriteBuffer) or isa(operation, host.TTEnqueueReadBuffer):
          self.print_ttenqueue_readwrite_buffer(operation)
        elif isa(operation, host.TTEnqueueProgram):
          self.print_ttenqueue_program(operation)
        elif isa(operation, host.TTFinish):
          self.print_ttfinish(operation)
        elif isa(operation, host.TTCloseDevice):
          self.print_ttclose_device(operation)
        elif isa(operation, host.TTSetRuntimeArgs):
          self.print_ttset_runtime_args(operation)
        elif type(operation) in TenstorrentOps:
            self.print_tt_op_generic(operation)

        #if isinstance(creator, CircularBufferOperationWithResult):
        #    arg1 = self.get_rhs_value(creator.operands[0])
        #    arg2 = self.get_rhs_value(creator.operands[1])
        #    return f"{creator.name.replace('.', '_')}({arg1}, {arg2})"

        else:
            raise NotImplementedError(f"Unhandled operation: {operation.__class__.__name__}")

    def print_return(self, op):
      if len(op.arguments) > 0:
        assert len(op.arguments)==1
        self.print("return ", indented=True)
        self.print_expr(op.arguments[0])
        self.print(";", end='\n')

    def print_ttset_runtime_args(self, op):
        self.print("SetRuntimeArgs(", indented=True)
        self.print_expr(op.program)
        self.print(", ")
        self.print_expr(op.kernel)
        self.print(", ")
        self.print_expr(op.core)
        self.print(", {")
        for idx, arg in enumerate(op.args):
          if idx > 0: self.print(", ")
          self.print_expr(arg)
        self.print("});", end='\n')

    def print_ttclose_device(self, op):
        self.print("CloseDevice(", indented=True)
        self.print_expr(op.device)
        self.print(");", end='\n')

    def print_ttfinish(self, op):
        self.print("Finish(", indented=True)
        self.print_expr(op.command_queue)
        self.print(");", end='\n')

    def print_ttenqueue_program(self, op):
        self.print("EnqueueProgram(", indented=True)
        self.print_expr(op.command_queue)
        self.print(", ")
        self.print_expr(op.program)
        self.print(", ")
        self.print_expr(op.blocking)
        self.print(");", end='\n')

    def print_ttenqueue_readwrite_buffer(self, op):
        if isa(op, host.TTEnqueueWriteBuffer):
          self.print("EnqueueWriteBuffer(", indented=True)
        elif isa(op, host.TTEnqueueReadBuffer):
          self.print("EnqueueReadBuffer(", indented=True)
        else:
          assert False
        self.print_expr(op.command_queue)
        self.print(", ")
        self.print_expr(op.buffer)
        self.print(", ")
        self.print_expr(op.data)
        self.print(", ")
        self.print_expr(op.blocking)
        self.print(");", end='\n')

    def print_expr(self, ssa_val):
      expr=ssa_val.owner
      if isa(expr, arith.ConstantOp):
        if isa(expr.result.type, builtin.IntegerType) and expr.result.type.width.data==1:
          self.print(f"{'false' if expr.value.value.data==0 else 'true'}")
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
      elif isa(expr, host.TTCreateBuffer):
          self.print_ttcreate_buffer(expr)
      elif isa(expr, host.TTCreateKernel):
          self.print_ttcreate_kernel(expr)
      elif isa(expr, host.TTGetMemoryAddress):
          self.print_ttget_memory_address(expr)
      elif isa(expr, memref.LoadOp):
          self.print_load_variable(expr)
      elif isa(expr, Block):
          self.print(f"fn_arg_{ssa_val.index}")
      elif isa(expr, memref.AllocaOp) or isa(expr, memref.AllocOp):
          self.print(expr.results[0].name_hint)
      elif isa(expr,BinaryOperation):
          self.print_binary_op(expr)
      elif isa(expr, arith.ExtFOp):
          self.print_cast_to_float(expr)
      elif isa(expr, arith.IndexCastOp):
          # Go directly to the operation used as an input and process this
          self.print_expr(expr.input)
      elif type(expr) in TenstorrentExpr:
          self.print_tt_expr_generic(expr)
      else:
        raise NotImplementedError(f"Unhandled expression: {expr.__class__.__name__}")

    def print_load_variable(self, op):
      self.print(op.memref.name_hint)
      if len(op.indices) > 0:
        # For now we limit ourselves to one dimensional arrays
        assert len(op.indices) == 1
        self.print("[")
        self.print_expr(op.indices[0])
        self.print("]")

    def print_cast_to_float(self, op):
        self.print("static_cast<float>(")
        self.print_expr(op.operands[0])
        self.print("(")

    def print_tthost_core(self, op):
        self.print("{")
        self.print_expr(op.src_noc_x)
        self.print(", ")
        self.print_expr(op.src_noc_y)
        self.print("}");

    def print_ttget_memory_address(self, op):
        self.print_expr(op.buffer)
        self.print("->address()")

    def print_ttcreate_kernel(self, op):
        self.print("CreateKernel(")
        self.print_expr(op.program)
        self.print(f", {op.kernel_name}, ")
        self.print_expr(op.core)
        self.print(", ")

        rv_core_flag=list(op.riscv_core.flags)[0]
        if rv_core_flag == host.RISCVCoreFlags.DATAMOVEMENT_0 or rv_core_flag == host.RISCVCoreFlags.DATAMOVEMENT_1:
          self.print("DataMovementConfig{.processor = DataMovementProcessor::RISCV_")
          if rv_core_flag == host.RISCVCoreFlags.DATAMOVEMENT_0:
            self.print("0")
          else:
            self.print("1")

          self.print(f", .noc=NOC::RISCV_{op.noc_id.data}_default}}")

        self.print(")")

    def print_ttcreate_buffer(self, op):
        self.print("CreateBuffer(")
        self.print_expr(op.config)
        self.print(")")

    def print_ttcreate_dram_config(self, op):
        self.print("{")
        self.print(".device=device")
        self.print(", .size=")
        self.print_expr(op.size)
        self.print(", .page_size=")
        self.print_expr(op.page_size)
        self.print(", .buffer_type = BufferType::DRAM")
        self.print("}");

        #.device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};

    def print_ttcreate_device(self, op):
        self.print("CreateDevice(")
        self.print_expr(op.index)
        self.print(")")

    def print_ttcreate_program(self, op):
        self.print("CreateProgram()")

    def print_ttget_command_queue(self, op):
        assert isa(op.device.owner, memref.LoadOp)
        self.print(f"{op.device.owner.memref.name_hint}->command_queue()")

    def print_binary_op(self, op):
      if isinstance(op, arith.ComparisonOperation):
        self.print_expr(op.lhs)

        assert op.predicate.value.data < len(CMP_PREDICATE_TO_SYMBOL)
        print(f" {CMP_PREDICATE_TO_SYMBOL[op.predicate.value.data]} ", end='')

        self.print_expr(op.rhs)
      else:
        if isa(op, arith.XOrIOp):
          self.print("!")

        self.print_expr(op.lhs)

        if not isa(op, arith.XOrIOp):
          self.print(f" {ARITH_OP_TO_SYM[op.__class__]} ", end='')

        self.print_expr(op.rhs)

    def print_func_def(self, func_op: func.FuncOp):
        """
        void func_name(typea a, typeb b, ...) {

        }
        """
        is_tt_kernel=self.is_tt_kernel(func_op)
        return_type="void"

        if len(func_op.function_type.outputs) > 0:
          assert len(func_op.function_type.outputs) == 1
          return_type = MLIR_TO_CPP_TYPES[func_op.function_type.outputs.data[0]]

        self.print(f"{return_type} {func_op.sym_name.data}(", indented=True)

        if not is_tt_kernel:
          for idx, input in enumerate(func_op.function_type.inputs):
            type_decl = MLIR_TO_CPP_TYPES[input]
            if idx > 0: self.print(", ")
            self.print(f"{type_decl} fn_arg_{idx}")

        self.print(") {", end='\n')

        self._indent += 1

        if is_tt_kernel:
          for idx, input in enumerate(func_op.function_type.inputs):
            type_decl = MLIR_TO_CPP_TYPES[input]
            self.print(f"{type_decl} fn_arg_{idx} = get_arg_val<{type_decl}>({idx});", indented=True, end='\n')

        # Have to do some messing around here with the return operation, which must be the last operation. That is
        # because we insert free for heap allocated memory, but that must be before the return, hence extract this,
        # print the rest, then print the free and lastly print the return
        ret_op=func_op.body.block.ops.last
        assert isa(ret_op, func.ReturnOp)

        self.print_region(list(func_op.body.block.ops)[:-1])

        for free_c in self._free_end_of_fn:
          self.print(f"free({free_c});", indented=True, end='\n');

        self.print_op(ret_op)

        self._indent -= 1

        self._free_end_of_fn=[]

        self.print("}", indented=True, end='\n')

    def is_tt_kernel(self, func):
      op=func.parent.parent.parent
      assert isa(op, builtin.ModuleOp)
      kernel_type=op.attributes["kernel_type"].data
      return kernel_type == "data_in" or kernel_type == "data_out"


    def print_for_loop(self, loop: scf.ForOp):
        # we know the first operation in the loop should be the store into i
        store_i = loop.body.block.first_op
        loop_index_name=store_i.operands[1].name_hint

        self.print(f"for ({loop_index_name}=", indented=True)
        self.print_expr(loop.lb)
        self.print(f";{loop_index_name}<")
        self.print_expr(loop.ub)
        self.print(f";{loop_index_name}+=")
        self.print_expr(loop.step)
        self.print(") {", end='\n')

        self._indent += 1
        # Remove the first operation as this assigns the loop variable and
        # we have handled this above already by looking into that operation and
        # extracting the name
        self.print_region(list(loop.body.ops)[1:])
        self._indent -= 1

        self.print("}", indented=True, end='\n')

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
          pass
        else:
          var_name = self.create_fresh_variable(op.results[0].name_hint)
          self._names[op.results[0]] = var_name
          type_decl = MLIR_TO_CPP_TYPES[op.result_types[0].element_type]

          if len(op.result_types[0].shape) == 0:
            self.print(type_decl + " " + var_name, indented=True)

            # We are grabbing the assignment RHS and doing the assign in the declaration here, this is
            # because sometimes in CPP you require that on the declaration (e.g. a reference)
            store_op_use=self.retrieve_store(op.results[0].uses)
            assert isa(store_op_use.operation, memref.StoreOp)
            if not isa(store_op_use.operation.operands[0], BlockArgument):
              self.print("=")
              self.print_expr(store_op_use.operation.operands[0])
              # A bit of a hack, we add this attribute to the store itself so that when this is
              # subsequently picked up by the assignment it can be ignored
              store_op_use.operation.attributes["ignore"]=True

            self.print(";", end='\n')
          else:
            total_size=1
            for s in op.result_types[0].shape:
              # Might not be correct for multi-dimensional arrays
              total_size*=s.data
            if isa(op, memref.AllocOp):
              self.print(f"{type_decl} * {var_name} =({type_decl}*) malloc(sizeof({type_decl})*{total_size});", indented=True, end='\n')
              self._free_end_of_fn.append(var_name)
            elif isa(op, memref.AllocaOp):
              self.print(f"{type_decl} {var_name}[{total_size}];", indented=True, end='\n')
            else:
              assert False

    def retrieve_store(self, uses):
      for use in uses:
        if isa(use.operation, memref.StoreOp): return use
      return None


    def print_assignment(self, op: memref.StoreOp):
        if "ignore" in op.attributes:
          if op.attributes["ignore"]: return
        ssa_value = op.operands[0]
        ssa_destination = op.operands[1]

        if isa(ssa_destination.type.element_type, host.DRAMBufferConfig):
          self.print(f"InterleavedBufferConfig {ssa_destination.name_hint} ", indented=True)

        else:
          var_name = self._names[ssa_destination]
          self.print(f"{var_name}", indented=True)
          if len(op.indices) > 0:
            # For now we limit ourselves to one dimensional arrays
            assert len(op.indices) == 1
            self.print("[")
            self.print_expr(op.indices[0])
            self.print("]")
          self.print(" = ")

        self.print_expr(ssa_value)
        self.print(";", end='\n')


    def print_if_statement(self, op: scf.IfOp):
        self.print("if (", indented=True)
        self.print_expr(op.cond)
        self.print(") {", end='\n')

        self._indent += 1
        self.print_op(op.true_region.blocks[0])
        self._indent -= 1

        or_else = len(op.false_region.blocks) > 0

        self.print("}" + (" else {" if or_else else ""), indented=True, end='\n')

        if or_else:
            self._indent += 1
            self.print_region(op.false_region)
            self._indent -= 1
            self.print("}", indented=True, end='\n')


    def create_fresh_variable(self, hint='a') -> str:
        names = self._names.values()
        if hint not in names:
            return hint

        count = 0
        name = hint
        while name in names:
            count += 1
            name = hint + str(count)

        return name

    def print_tt_expr_generic(self, expression):
      self.print_tt_operation_generic(expression, False)

    def print_tt_op_generic(self, operation):
      self.print_tt_operation_generic(operation, True)

    def print_tt_operation_generic(self, operation, is_expression):
        api_name = get_api_name(operation.name)
        self.print(api_name, indented=is_expression)
        if operation.properties:
          self.print("<")
          for idx, p in enumerate(operation.properties.values()):
            if idx > 0: self.print(", ")
            self.print(str(p))
          self.print(">")
        self.print("(")
        for idx, expr in enumerate(operation.operands):
          if idx > 0: self.print(", ")
          self.print_expr(expr)
        self.print(")")
        if is_expression:
          self.print(";", end='\n')


    def print(self, s: str, indented: bool = False, end=''):
        prefix = self._prefix if indented else ""
        if self._file:
            print(prefix + s, file=self._file, end=end)
        else:
            print(prefix + s, end=end)

    @property
    def _prefix(self):
        return " " * 4 * self._indent

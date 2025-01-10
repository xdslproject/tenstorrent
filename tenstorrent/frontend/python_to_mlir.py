import ast

from xdsl.dialects import builtin, func, arith, memref, scf
from xdsl.dialects.builtin import (IntegerAttr, IntegerType, FunctionType,
                                   MemRefType, ModuleOp, IndexType, FloatAttr, Float32Type)
from xdsl.ir import Operation, Region, Block, OpResult, SSAValue
from typing import Dict, List
from xdsl.irdl import IRDLOperation

from .memref_context import MemrefContext
from .type_checker import MLIRType
from .dummy import *
from xdsl.utils.hints import isa
from tenstorrent.utils import flatten, remove_duplicates, subtract
from tenstorrent.dialects import *

NodeWithBody = ast.If | ast.For | ast.While


class PythonToMLIR(ast.NodeVisitor):
    """
    Parses a Python AST to create operations from the tt_data, tt_compute,
    scf, and memref dialects.
    """

    def __init__(self, type_checker):
        super().__init__()
        self.symbol_table = MemrefContext()  # variable names -> memref

        self.operations: List[Operation] | ModuleOp = []
        self.type_checker = type_checker

        self._uint_comparison: Dict[type, str] = {
            ast.Eq: arith.CMPI_COMPARISON_OPERATIONS[0],
            ast.NotEq: arith.CMPI_COMPARISON_OPERATIONS[1],
            ast.Lt: arith.CMPI_COMPARISON_OPERATIONS[6],
            ast.LtE: arith.CMPI_COMPARISON_OPERATIONS[7],
            ast.Gt: arith.CMPI_COMPARISON_OPERATIONS[8],
            ast.GtE: arith.CMPI_COMPARISON_OPERATIONS[9],
        }

        self._sint_comparison: Dict[type, str] = {
            ast.Eq: arith.CMPI_COMPARISON_OPERATIONS[0],
            ast.NotEq: arith.CMPI_COMPARISON_OPERATIONS[1],
            ast.Lt: arith.CMPI_COMPARISON_OPERATIONS[2],
            ast.LtE: arith.CMPI_COMPARISON_OPERATIONS[3],
            ast.Gt: arith.CMPI_COMPARISON_OPERATIONS[4],
            ast.GtE: arith.CMPI_COMPARISON_OPERATIONS[5],
        }

        # use only ordered comparisons (don't allow NaN)
        self._float_comparison: Dict[type, str] = {
            ast.Eq: arith.CMPF_COMPARISON_OPERATIONS[1],
            ast.Gt: arith.CMPF_COMPARISON_OPERATIONS[2],
            ast.GtE: arith.CMPF_COMPARISON_OPERATIONS[3],
            ast.Lt: arith.CMPF_COMPARISON_OPERATIONS[4],
            ast.LtE: arith.CMPF_COMPARISON_OPERATIONS[5],
            ast.NotEq: arith.CMPF_COMPARISON_OPERATIONS[6],
        }

        self._operations: Dict[MLIRType, Dict[type(ast.operator), type(IRDLOperation)]] = {
            IntegerType(32): {
                ast.Add: arith.AddiOp,
                ast.Mult: arith.MuliOp,
                ast.Sub: arith.SubiOp,
                ast.Div: arith.DivfOp,
            },
            Float32Type(): {
                ast.Add: arith.AddfOp,
                ast.Mult: arith.MulfOp,
                ast.Sub: arith.SubfOp,
                ast.Div: arith.DivfOp,
            }
        }

        self._functions: Dict[str, type] = {
            cb_push_back.__name__: CBPushBack,
            cb_reserve_back.__name__: CBReserveBack,
            cb_pop_front.__name__: CBPopFront,
            cb_wait_front.__name__: CBWaitFront,
            cb_pages_reservable_at_back.__name__: CBPagesReservableAtBack,
            cb_pages_available_at_front.__name__: CBPagesAvailableAtFront,
            noc_async_read.__name__: DMNocAsyncRead,
            noc_async_write.__name__: DMNocAsyncWrite,
            noc_semaphore_set.__name__: DMNocSemaphoreSet,
            noc_semaphore_set_multicast.__name__: DMNocSemaphoreSetMulticast,
            noc_async_write_multicast.__name__: DMNocAsyncWriteMulticast,
            noc_semaphore_wait.__name__: DMNocSemaphoreWait,
            noc_semaphore_inc.__name__: DMNocSemaphoreInc,
            noc_async_read_barrier.__name__: DMNocAsyncReadBarrier,
            noc_async_write_barrier.__name__: DMNocAsyncWriteBarrier,
        }

    def get_type(self, variable_name: str):
        return self.type_checker.types[variable_name]

    def get_operation(self, node) -> type:
        assert isinstance(node, ast.BinOp)
        t = self.type_checker.visit(node)
        return self._operations[t][type(node.op)]

    def generic_visit(self, node):
        print("Missing handling for: " + node.__class__.__name__)
        raise Exception(f"Unhandled construct, no parser provided: {node.__class__.__name__}")

    def visit_Import(self, node) -> List[Operation]:
        return []

    def visit_Pass(self, node) -> List[Operation]:
        return []

    def visit_Module(self, node) -> List[Operation]:
        operations: List[Operation] = []

        # at the 'module' level should just be a single function def
        for child in node.body:
            ops = self.visit(child)
            operations.extend(ops)

        # after all processing is done, wrap in module operation
        self.operations = builtin.ModuleOp(operations)
        return [self.operations]

    def visit_FunctionDef(self, node) -> List[Operation]:
        operations: List[Operation] = []

        decorator_name=None
        if (len(node.decorator_list) == 1):
          decorator_name=(node.decorator_list[0].attr)

        # set the current scope
        self.operations = operations

        for child in node.body:
            ops, ssa = self.visit(child)
            operations+=ops

        operations.append(func.ReturnOp())

        block = Block(operations)
        region = Region(block)

        fn_name=node.name
        if decorator_name == "data_in":
          fn_name="kernel_main"
        elif decorator_name == "host":
          fn_name="main"

        func_op=func.FuncOp(
            "kernel_main",
            FunctionType.from_lists([], []),
            region
        )

        # return some function definition with contents
        return [builtin.ModuleOp([func_op], {"kernel_type": builtin.StringAttr(decorator_name)})]

    def visit_Assign(self, node) -> List[Operation]:
        # visit RHS, e.g. Constant in 'a = 0'
        rhs_ops, rhs_ssa_val = self.visit(node.value)
        operations = rhs_ops

        # get the variable name to store the result in
        dest = node.targets[0]
        if isa(dest, ast.Name):
          # Scalar assignment
          var_name = dest.id
        elif isa(dest, ast.Subscript):
          # Array assignment
          var_name = dest.value.id
        else:
          assert False

        # if the types don't match we need to insert a cast operation
        target_type = self.type_checker.types[var_name]
        if target_type != rhs_ssa_val.type:
            cast_ops, cast_ssa = self.get_cast(target_type, rhs_ssa_val)
            if len(cast_ops) > 0:
              assert cast_ssa is not None
              rhs_ssa_val = cast_ssa
              operations += cast_ops
            else:
              assert cast_ssa is None

        if isa(dest, ast.Name):
          # create a memref
          # seen = var_name in self.symbol_table
          seen = var_name in self.symbol_table.dictionary
          if isa(rhs_ops[-1], memref.AllocaOp):
            # This is a bit of a hack, we are allocating a list and this is done
            # in the bin op, so we pick the memref here
            self.symbol_table[var_name] = rhs_ops[-1].results[0]
            assert seen == False
            location = self.symbol_table[var_name]
          else:
            # This is a normal value, so just store as usual
            if not seen:
              operations += [self.allocate_memory(var_name)]

            location = self.symbol_table[var_name]

            # store result in that memref
            store = memref.StoreOp.get(rhs_ssa_val, location, [])
            operations.append(store)
          return operations, location
        elif isa(dest, ast.Subscript):
          # Array assignment
          assert isa(dest.slice, ast.Constant)
          idx_ops, idx_ssa=self.visit(dest.slice)
          assert var_name in self.symbol_table.dictionary
          store = memref.StoreOp.get(rhs_ssa_val, self.symbol_table[var_name], [idx_ssa])
          return operations + idx_ops + [store], self.symbol_table[var_name]

    def get_cast(self, target_type: MLIRType, ssa_val: SSAValue) -> Operation:
        if isinstance(ssa_val.type, IntegerType):
            conv_op=None
            match target_type:
                case Float32Type():
                    conv_op=arith.ExtFOp(ssa_val, Float32Type())

                case IndexType():
                    conv_op=arith.IndexCastOp(ssa_val, IndexType())
            if conv_op is not None:
              return [conv_op], conv_op.results[0]
        return [], None


    def visit_Constant(self, node):
        data = node.value

        arith_op=None

        if isinstance(data, bool):
            arith_op = arith.ConstantOp(IntegerAttr(data, IntegerType(1)))
        elif isinstance(data, int):
            arith_op = arith.ConstantOp(IntegerAttr(data, IntegerType(32)))
        elif isinstance(data, float):
            arith_op = arith.ConstantOp(FloatAttr(data, Float32Type()))

        if arith_op is None:
          raise Exception(f"Unhandled constant type: {data.__class__.__name__}")

        return [arith_op], arith_op.results[0]

    def visit_List(self, node: ast.List):
      assert len(node.elts) == 1
      element_type_ops, element_type_ssa=self.visit(node.elts[0])
      return None, MemRefType(element_type_ssa.type, [])

    def visit_BinOp(self, node: ast.BinOp) -> List[Operation]:
        lhs_ops, lhs_ssa_val = self.visit(node.left)
        rhs_ops, rhs_ssa_val = self.visit(node.right)

        if isa(lhs_ssa_val, MemRefType):
          # This is a list definition, grab the size on the RHS and use memref alloc
          if isa(rhs_ops[0], arith.ConstantOp):
            assert isa(rhs_ops[0].value.type, IntegerType)
            size_val=rhs_ops[0].value.value.data
            alloc_shape=[size_val]
            dynamic_size=[]
          else:
            alloc_shape=[-1]
            dynamic_size=[rhs_ssa_val]

          memory = memref.AllocaOp.get(lhs_ssa_val.element_type, shape=alloc_shape, dynamic_sizes=dynamic_size)
          operations = rhs_ops + [memory]
          return operations, memory.results[0]
        else:
          operations = lhs_ops + rhs_ops

          # if types differ, we need to cast for the operation
          if lhs_ssa_val.type != rhs_ssa_val.type:
              target_type = self.type_checker.dominating_type(lhs_ssa_val.type, rhs_ssa_val.type)
              if lhs_ssa_val.type != target_type:
                  l_cast = self.get_cast(target_type, lhs_ssa_val)
                  operations += [l_cast]
                  lhs_ssa_val = l_cast.results[0]

              if rhs_ssa_val.type != target_type:
                  r_cast = self.get_cast(target_type, rhs_ssa_val)
                  operations += [r_cast]
                  rhs_ssa_val = r_cast.results[0]

          # special case: if we have a division, we also want to cast
          if isinstance(node, ast.Div):
              target_type = Float32Type()
              if lhs_ssa_val.type != target_type:
                  l_cast = self.get_cast(target_type, lhs_ssa_val)
                  operations += [l_cast]
                  lhs_ssa_val = l_cast.results[0]

              if rhs_ssa_val.type != target_type:
                  r_cast = self.get_cast(target_type, rhs_ssa_val)
                  operations += [r_cast]
                  rhs_ssa_val = r_cast.results[0]

          op_constructor = self.get_operation(node)
          bin_op=op_constructor(
              lhs_ssa_val,
              rhs_ssa_val,
              None
          )
          return operations + [bin_op], bin_op.results[0]

    def visit_Name(self, node: ast.Name) -> List[Operation]:
        from_location = self.symbol_table[node.id]
        assert isa(from_location.type, MemRefType)
        if (len(from_location.type.shape) == 0):
          # If this is a scalar then load it from the memref
          load = memref.LoadOp.get(from_location, [])
          return [load], load.results[0]
        else:
          # If it is an array then return directly as we don't have array indexes
          # because if we did then this would be a subscript instead
          return [], from_location


    def visit_If(self, node: ast.If) -> List[Operation]:
        allocations = self.allocate_new_variables(node)
        body_ops = self.generate_body_ops(node)

        condition_expr_ops, condition_ssa_val = self.visit(node.test)

        #or_else = None
        #if node.orelse:
        #    if isinstance(node.orelse[0], ast.If):
        #        or_else_ops, or_else_ssa_val = self.visit_If(node.orelse[0])
        #    else:
        #        or_else = []
        #        for stmt in node.orelse:
        #            or_else += self.visit(stmt)

        # condition: SSAValue | Operation
        # return_types: Sequence[Attribute],
        # true_region: Region | Sequence[Block] | Sequence[Operation]
        # false_region: Region | Sequence[Block] | Sequence[Operation]
        if_statement = scf.If(
            condition_ssa_val,
            [],
            body_ops,
            [] #or_else
        )

        return allocations + condition_expr_ops + [condition, if_statement], if_statement.results[0]


    def visit_Compare(self, node) -> List[Operation]:
        left = self.visit(node.left)[0]
        right = self.visit(node.comparators[0])[0]

        l_val = left.results[0]
        r_val = right.results[0]

        # TODO: handle sint, float comparisons
        op = self._uint_comparison[type(node.ops[0])]
        operation = arith.Cmpi(l_val, r_val, op)

        return [left, right, operation], operation.results[0]


    def visit_For(self, node: ast.For) -> List[Operation]:
        from_expr = self.visit(node.iter.args[0])[-1]
        to_expr = self.visit(node.iter.args[1])[-1]

        # lb, up, step, iteration arguments, body
        step = arith.Constant(IntegerAttr(1, IntegerType(32)))

        # adds variables to the symbol table and allocates memory for them
        var_allocations = self.allocate_new_variables(node)

        block = Block(arg_types=[IndexType()])

        body = Region()
        body.add_block(block)

        for_loop = scf.For(
            from_expr.results[0],
            to_expr.results[0],
            step.results[0],
            [],
            body
        )

        # if going for correct python semantics should probably manually
        # allocate space for the loop variable, and load into it for the first
        # instruction of the loop.
        iter_var_name = node.target.id
        alloc = self.allocate_memory(iter_var_name)

        loop_variable = for_loop.body.block.args[0]
        store = memref.Store.get(loop_variable, alloc, [])

        loop_body_ops = [store] + self.generate_body_ops(node) + [scf.Yield()]
        block.add_ops(loop_body_ops)

        return var_allocations + [from_expr, to_expr, step, alloc, for_loop], for_loop.results[0]


    def visit_BoolOp(self, node: ast.BoolOp) -> List[Operation]:
        # leftmost evaluation first
        # if a and b or c => if (a and b) or c
        lhs_ops, lhs_ssa_val = self.visit(node.values[0])   # a and b
        rhs_ops, rhs_ssa_val = self.visit(node.values[1])  # c

        match type(node.op):
            case ast.And:
                op = arith.AndI

            case ast.Or:
                op = arith.OrI

            case _:
                raise NotImplementedError(f"{node.op.__class__.__name__}")

        operation = op(
            lhs_ssa_val,
            rhs_ssa_val
        )

        return lhs_ops + rhs_ops + [operation], operation.results[0]

    def visit_UnaryOp(self, node) -> List[Operation]:
        expr_ops, expr_ssa_val = self.visit(node.operand)
        true_decl = arith.Constant(IntegerAttr.from_int_and_width(1, 1))

        match type(node.op):
            case ast.Not:
                unary_op = arith.XOrI(expr_ssa_val, true_decl.results[0])
                return expr_ops + [
                    true_decl,
                    unary_op], unary_op.results[0]
            case ast.USub:
                zero = arith.Constant(IntegerAttr(0, IntegerType(32)))
                unary_op = arith.Subi(zero.results[0], expr_ssa_val)
                return expr_ops + [
                    zero,
                    unary_op], unary_op.results[0]
            case _:
                raise NotImplementedError(f"{node.op.__class__.__name__}")


    def visit_Expr(self, node) -> List[Operation]:
        return self.visit(node.value)

    def handleCreateKernel(self, node):
      assert len(node.args)==5

      program_ops, program_ssa=self.visit(node.args[0])
      core_ops, core_ssa=self.visit(node.args[2])

      assert isa(node.args[1], ast.Name)
      target_fn_name=node.args[1].id

      assert isa(node.args[3], ast.Attribute)

      rv_core_flag=None
      if node.args[3].attr == "DataMovement_0":
        rv_core_flag=RISCVCoreFlags.DATAMOVEMENT_0
      elif node.args[3].attr == "DataMovement_1":
        rv_core_flag=RISCVCoreFlags.DATAMOVEMENT_1
      elif node.args[3].attr == "Compute":
        rv_core_flag=RISCVCoreFlags.COMPUTE

      assert rv_core_flag is not None

      assert isa(node.args[4], ast.Constant)
      noc_id=node.args[4].value
      assert noc_id == 0 or noc_id == 1

      kernelCreate=TTCreateKernel(program_ssa, core_ssa, target_fn_name+"_kernel.cpp", RISCVCoreFlagsAttr([rv_core_flag]), noc_id)

      return program_ops + core_ops + [kernelCreate], kernelCreate.results[0]


    def handleHostCall(self, node, operationClass, expectedNumArgs):
      if expectedNumArgs is not None: assert len(node.args)==expectedNumArgs
      arg_ops=[]
      arg_ssas=[]
      for arg in node.args:
        ops, ssa=self.visit(arg)
        arg_ops+=ops
        arg_ssas.append(ssa)

      operation=operationClass(*arg_ssas)
      if len(operation.results) > 0:
        return arg_ops+[operation], operation.results[0]
      else:
        return arg_ops+[operation], None


    def visit_Call(self, node) -> List[Operation]:
        if isa(node.func, ast.Attribute):
          name=node.func.attr
          if name == "Core": return self.handleHostCall(node, TTHostCore, 2)
          if name == "DRAMConfig": return self.handleHostCall(node, TTCreateDRAMConfig, 2)
          if name == "CreateBuffer": return self.handleHostCall(node, TTCreateBuffer, 1)
          if name == "CreateDevice": return self.handleHostCall(node, TTCreateDevice, 1)
          if name == "GetCommandQueue": return self.handleHostCall(node, TTGetCommandQueue, 1)
          if name == "EnqueueWriteBuffer": return self.handleHostCall(node, TTEnqueueWriteBuffer, 4)
          if name == "EnqueueReadBuffer": return self.handleHostCall(node, TTEnqueueReadBuffer, 4)
          if name == "CreateProgram": return self.handleHostCall(node, TTCreateProgram, 0)
          if name == "Kernel": return self.handleCreateKernel(node)
          if name == "SetRuntimeArgs": return self.handleHostCall(node, TTSetRuntimeArgs, None)
          if name == "EnqueueProgram": return self.handleHostCall(node, TTEnqueueProgram, 3)
          if name == "Finish": return self.handleHostCall(node, TTFinish, 1)
          if name == "CloseDevice": return self.handleHostCall(node, TTCloseDevice, 1)
        else:
          name = node.func.id
          if name not in self._functions:
              raise NotImplementedError(f"Unhandled function {name}")

        # We evaluate args in Python order (programmer intention) and then swap
        # only the SSA results that are given to the operation to preserve semantics
        ops_per_arg = [self.visit(arg) for arg in node.args]
        operations = list(flatten(ops_per_arg))
        results = list(map(lambda ops: ops[-1].results[0], ops_per_arg))

        match name:
            case noc_async_write_multicast.__name__:
                results[4], results[5], results[6] = results[5], results[6], results[4]
            case noc_semaphore_set_multicast.__name__:
                results[3], results[4], results[5] = results[4], results[5], results[3]

        operation = self._functions[name](*results)
        return operations + [operation], operation.results[0]


    def generate_body_ops(self, node: NodeWithBody) -> List[Operation]:
        return [op for statement in node.body for op in self.visit(statement)]


    def get_assigned_variables(self, statement: ast.stmt) -> List[str]:
        if isinstance(statement, ast.Assign):
            return [statement.targets[0].id]

        if isinstance(statement, ast.For | ast.If | ast.While):
            names = []

            for child_stmt in statement.body:
                names += self.get_assigned_variables(child_stmt)

            return names

        # could also handle: ast.With, ast.FuncDef
        construct = statement.__class__.__name__
        raise Exception(f"Unhandled construct to explore: {construct}")


    def allocate_new_variables(self, node: NodeWithBody) -> List[Operation]:
        """
        In Python, variables declared/initialised in the loop body persist the
        loop itself - there is no nested scope. This method searches a loop or
        if statement body in order to find all these variables and allocate them
        before the scf.For (etc.) scope. Sets preserve order in CPython 3.7+.
        """
        found_variables = []
        for statement in node.body:
            found_variables += self.get_assigned_variables(statement)

        # remove any existing variables from fresh variables
        found_variables = remove_duplicates(found_variables)
        fresh_variables = subtract(found_variables, items=self.symbol_table.dictionary)

        allocations = []
        for var in list(fresh_variables):
            memory = self.allocate_memory(var)
            allocations.append(memory)

        return allocations

    def allocate_memory(self, symbol: str) -> memref.AllocOp:
        memory = memref.AllocOp([], [], MemRefType(self.get_type(symbol), []))
        self.symbol_table[symbol] = memory.results[0]
        return memory

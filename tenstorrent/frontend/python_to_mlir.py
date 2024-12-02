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
                ast.Add: arith.Addi,
                ast.Mult: arith.Muli,
                ast.Sub: arith.Subi,
                ast.Div: arith.Divf,
            },
            Float32Type(): {
                ast.Add: arith.Addf,
                ast.Mult: arith.Mulf,
                ast.Sub: arith.Subf,
                ast.Div: arith.Divf,
            }
        }

        self._functions: Dict[str, type] = {
            cb_push_back.__name__: CBPushBack,
            cb_reserve_back.__name__: CBReserveBack,
            cb_pop_front.__name__: CBPopFront,
            cb_wait_front.__name__: CBWaitFront,
            cb_pages_reservable_at_back.__name__: CBPagesReservableAtBack,
            cb_pages_available_at_front.__name__: CBPagesAvailableAtFront,
        }

    def get_type(self, variable_name: str):
        return self.type_checker.types[variable_name]

    def visit(self, node: ast.AST | ast.stmt | ast.expr) -> List[Operation]:
        return [ob for ob in flatten(super().visit(node))]

    def get_operation(self, node) -> type:
        assert isinstance(node, ast.BinOp)
        t = self.type_checker.visit(node)
        return self._operations[t][type(node.op)]

    def generic_visit(self, node):
        print("Missing handling for: " + node.__class__.__name__)
        raise Exception(f"Unhandled construct, no parser provided: {node.__class__.__name__}")

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

        # set the current scope
        self.operations = operations

        for child in node.body:
            ops = self.visit(child)
            operations.extend(ops)

        block = Block(operations)
        region = Region(block)

        # return some function definition with contents
        return [func.FuncOp(
            node.name,
            FunctionType.from_lists([], []),
            region
        )]

    def visit_Assign(self, node) -> List[Operation]:
        # visit RHS, e.g. Constant in 'a = 0'
        rhs = self.visit(node.value)
        rhs_val = rhs[-1].results[0]
        operations = rhs

        # get the variable name to store the result in
        dest = node.targets[0]
        assert isinstance(dest, ast.Name)
        var_name = dest.id

        # create a memref
        # seen = var_name in self.symbol_table
        seen = var_name in self.symbol_table.dictionary
        location = self.allocate_memory(var_name) if not seen else self.symbol_table[var_name]

        # if the types don't match we need to insert a cast operation
        target_type = self.type_checker.types[var_name]
        if target_type != rhs_val.type:
            cast = self.get_cast(target_type, rhs_val)
            rhs_val = cast.results[0]
            operations += [cast]

        if not seen:
            operations += [location]

        # store result in that memref
        store = memref.Store.get(rhs_val, location, [])

        return operations + [store]

    def get_cast(self, target_type: MLIRType, ssa_val: SSAValue) -> Operation:
        if isinstance(ssa_val.type, IntegerType):
            match target_type:
                case Float32Type():
                    return arith.ExtFOp(ssa_val, Float32Type())

                case IndexType():
                    return arith.IndexCastOp(ssa_val, IndexType())

        raise NotImplementedError(f"Casting from {ssa_val.type.__class__.__name__} "
                                  f"to {target_type.__class__.__name__}")


    def visit_Constant(self, node) -> List[Operation]:
        data = node.value

        if isinstance(data, bool):
            return [arith.Constant(IntegerAttr(data, IntegerType(1)))]

        if isinstance(data, int):
            return [arith.Constant(IntegerAttr(data, IntegerType(32)))]

        if isinstance(data, float):
            return [arith.Constant(FloatAttr(data, Float32Type()))]

        raise Exception(f"Unhandled constant type: {data.__class__.__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> List[Operation]:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)

        # need to insert operations at evaluation order:
        for op in lhs + rhs:
            if op not in self.operations:
                self.operations.append(op)

        operations = []

        # get references to the SSA values
        l_val: OpResult = lhs[-1].results[0]
        r_val: OpResult = rhs[-1].results[0]

        # if types differ, we need to cast for the operation
        if l_val.type != r_val.type:
            target_type = self.type_checker.dominating_type(l_val.type, r_val.type)
            if l_val.type != target_type:
                l_cast = self.get_cast(target_type, l_val)
                operations += [l_cast]
                l_val = l_cast.results[0]

            if r_val.type != target_type:
                r_cast = self.get_cast(target_type, r_val)
                operations += [r_cast]
                r_val = r_cast.results[0]

        # special case: if we have a division, we also want to cast
        if isinstance(node, ast.Div):
            target_type = Float32Type()
            if l_val.type != target_type:
                l_cast = self.get_cast(target_type, l_val)
                operations += [l_cast]
                l_val = l_cast.results[0]

            if r_val.type != target_type:
                r_cast = self.get_cast(target_type, r_val)
                operations += [r_cast]
                r_val = r_cast.results[0]

        op_constructor = self.get_operation(node)
        return operations + [op_constructor(
            l_val,
            r_val,
            None
        )]

    def visit_Name(self, node: ast.Name) -> List[Operation]:
        from_location = self.symbol_table[node.id]
        load = memref.Load.get(from_location, [])
        return [load]


    def visit_If(self, node: ast.If) -> List[Operation]:
        allocations = self.allocate_new_variables(node)
        body_ops = self.generate_body_ops(node)

        condition_expr = self.visit(node.test)
        condition = condition_expr.pop()

        or_else = None
        if node.orelse:
            if isinstance(node.orelse[0], ast.If):
                or_else = self.visit_If(node.orelse[0])
            else:
                or_else = []
                for stmt in node.orelse:
                    or_else += self.visit(stmt)

        # condition: SSAValue | Operation
        # return_types: Sequence[Attribute],
        # true_region: Region | Sequence[Block] | Sequence[Operation]
        # false_region: Region | Sequence[Block] | Sequence[Operation]
        if_statement = scf.If(
            condition,
            [],
            body_ops,
            or_else
        )

        return allocations + condition_expr + [condition, if_statement]


    def visit_Compare(self, node) -> List[Operation]:
        left = self.visit(node.left)[0]
        right = self.visit(node.comparators[0])[0]

        l_val = left.results[0]
        r_val = right.results[0]

        # TODO: handle sint, float comparisons
        op = self._uint_comparison[type(node.ops[0])]
        operation = arith.Cmpi(l_val, r_val, op)

        return [left, right, operation]


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

        return var_allocations + [from_expr, to_expr, step, alloc, for_loop]


    def visit_BoolOp(self, node: ast.BoolOp) -> List[Operation]:
        # leftmost evaluation first
        # if a and b or c => if (a and b) or c
        left_ops = self.visit(node.values[0])   # a and b
        right_ops = self.visit(node.values[1])  # c

        match type(node.op):
            case ast.And:
                op = arith.AndI

            case ast.Or:
                op = arith.OrI

            case _:
                raise NotImplementedError(f"{node.op.__class__.__name__}")

        operation = op(
            left_ops[-1].results[0],
            right_ops[-1].results[0]
        )

        return left_ops + right_ops + [operation]

    def visit_UnaryOp(self, node) -> List[Operation]:
        expr = self.visit(node.operand)
        true_decl = arith.Constant(IntegerAttr.from_int_and_width(1, 1))

        match type(node.op):
            case ast.Not:
                return expr + [
                    true_decl,
                    arith.XOrI(expr[-1], true_decl.results[0])
                ]
            case ast.USub:
                zero = arith.Constant(IntegerAttr(0, IntegerType(32)))
                return expr + [
                    zero,
                    arith.Subi(zero.results[0], expr[-1].results[0])
                ]
            case _:
                raise NotImplementedError(f"{node.op.__class__.__name__}")


    def visit_Expr(self, node) -> List[Operation]:
        return self.visit(node.value)


    def visit_Call(self, node) -> List[Operation]:
        name = node.func.id
        if name in self._functions:
            arg1 = self.visit(node.args[0])
            arg2 = self.visit(node.args[1])

            operation = self._functions[name](
                arg1[-1].results[0],
                arg2[-1].results[0]
            )
            return arg1 + arg2 + [operation]

        raise NotImplementedError(f"Unhandled function {name}")


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

    def allocate_memory(self, symbol: str) -> memref.Alloc:
        memory = memref.Alloc([], [], MemRefType(self.get_type(symbol), [1]))
        self.symbol_table[symbol] = memory
        return memory

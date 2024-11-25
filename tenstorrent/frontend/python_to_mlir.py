import ast

from xdsl.dialects import builtin, func, arith, memref, scf
from xdsl.dialects.builtin import (IntegerAttr, IntegerType, FunctionType,
                                   MemRefType, ModuleOp, IndexType, FloatAttr, Float32Type)
from xdsl.ir import Operation, Region, Block, OpResult
from typing import Dict, List
from xdsl.irdl import IRDLOperation

from .memref_context import MemrefContext
from tenstorrent.utils import flatten, remove_duplicates

NodeWithBody = ast.If | ast.For | ast.While
MLIRType = IntegerType | Float32Type


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

        self._operations: Dict[MLIRType, Dict[type(ast.operator), type(IRDLOperation)]] = {
            IntegerType(32): {
                ast.Add: arith.Addi,
                ast.Mult: arith.Muli
            },
            Float32Type(): {
                ast.Add: arith.Addf,
                ast.Mult: arith.Mulf
            }
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
        rhs: Operation = self.visit(node.value)[0]

        # get the variable name to store the result in
        dest = node.targets[0]
        assert isinstance(dest, ast.Name)
        var_name = dest.id

        # create a memref
        # seen = var_name in self.symbol_table
        seen = var_name in self.symbol_table.dictionary
        location = memref.Alloc(
            [],
            [],  # symbol operands: [SSAValue]
            MemRefType(self.get_type(var_name), [1])
        ) if not seen else self.symbol_table[var_name]

        # store result in that memref
        store = memref.Store.get(rhs, location, [])
        self.register_symbol(var_name, location)

        # want to return whatever we wish to insert in our list of operations
        # return (rhs, store) if seen else (rhs, location, store)
        if seen:
            return [rhs, store]

        return [rhs, location, store]

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
        lhs: Operation = self.visit(node.left)[0]  # operation that created a, same operation returned from visit_Assign
        rhs: Operation = self.visit(node.right)[0]

        if lhs not in self.operations:
            self.operations.append(lhs)

        if rhs not in self.operations:
            self.operations.append(rhs)

        # get references to the SSA values
        l_val: OpResult = lhs.results[0]
        r_val: OpResult = rhs.results[0]

        op_constructor = self.get_operation(node)
        return [op_constructor(
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
        op = node.ops[0]

        l_val = left.results[0]
        r_val = right.results[0]

        equals = arith.CMPI_COMPARISON_OPERATIONS[0]
        operation = arith.Cmpi(l_val, r_val, equals)

        return [left, right, operation]


    def visit_For(self, node: ast.For) -> List[Operation]:
        from_expr = self.visit(node.iter.args[0])[0]
        to_expr = self.visit(node.iter.args[1])[0]

        start = arith.IndexCastOp(from_expr, IndexType())
        end = arith.IndexCastOp(to_expr, IndexType())

        # lb, up, step, iteration arguments, body
        step = arith.Constant.create(
            properties={
                "value": IntegerAttr.from_index_int_value(1)
            },
            result_types=[IndexType()]
        )

        # adds variables to the symbol table and allocates memory for them
        var_allocations = self.allocate_new_variables(node)

        block = Block(arg_types=[IndexType()])

        body = Region()
        body.add_block(block)

        for_loop = scf.For(
            start.results[0],
            end.results[0],
            step.results[0],
            [],
            body
        )

        # if going for correct python semantics should probably manually
        # allocate space for the loop variable, and load into it for the first
        # instruction of the loop.
        iter_var_name = node.target.id
        alloc = memref.Alloc([], [], MemRefType(self.get_type(iter_var_name), [1]))
        self.symbol_table[iter_var_name] = alloc

        loop_variable = for_loop.body.block.args[0]
        store = memref.Store.get(loop_variable, alloc, [])

        loop_body_ops = [store] + self.generate_body_ops(node) + [scf.Yield()]
        block.add_ops(loop_body_ops)

        return var_allocations + [from_expr, to_expr, start, end, step, alloc, for_loop]


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
                raise NotImplementedError(f"BoolOp {node.op.__class__.__name__}")

        operation = op(
            left_ops[-1].results[0],
            right_ops[-1].results[0]
        )

        return left_ops + right_ops + [operation]


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

        found_variables = remove_duplicates(found_variables)

        # remove any existing variables from fresh variables
        fresh_variables = []
        for var_name in found_variables:
            if var_name not in self.symbol_table.dictionary:
                fresh_variables.append(var_name)

        allocations = []

        for var in list(fresh_variables):
            memory = memref.Alloc(
                [],
                [],
                MemRefType(self.get_type(var), [1])
            )
            self.register_symbol(var, memory)
            allocations.append(memory)

        return allocations


    def register_symbol(self, symbol: str, memory: memref.Alloc):
        self.symbol_table[symbol] = memory

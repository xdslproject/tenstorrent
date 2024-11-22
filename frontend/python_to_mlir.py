import ast

from xdsl.dialects import builtin, func, arith, memref, scf
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import IntegerAttr, IntegerType, FunctionType, MemRefType, ModuleOp, IndexType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import Store
from xdsl.ir import Operation, Region, Block, SSAValue, OpResult, BlockArgument
from typing import Dict, List, Optional, cast, Tuple, Set

from xdsl.irdl import IRDLOperation

from frontend.memref_context import MemrefContext
from frontend.util import flatten, remove_duplicates

NodeWithBody = ast.If | ast.For | ast.While


class PythonToMLIR(ast.NodeVisitor):
    """
    Parses a Python AST to create operations from the tt_data, tt_compute,
    scf, and memref dialects.
    """

    def __init__(self):
        super().__init__()
        self.symbol_table = MemrefContext()  # variable names -> memref

        self.operations: List[Operation] | ModuleOp = []

        self._operations: Dict[bool, Dict[type(ast.operator), type(IRDLOperation)]] = {
            False: {
                ast.Add: arith.Addi,
                ast.Mult: arith.Muli
            },
            True: {
                ast.Add: arith.Addf,
                ast.Mult: arith.Mulf
            }
        }

    def visit(self, node: ast.AST | ast.stmt | ast.expr) -> List[Operation]:
        return [ob for ob in flatten(super().visit(node))]

    def get_operation(self, node) -> type:
        assert isinstance(node, ast.BinOp)
        # TODO: will need to check if either child is floating point (recursively)
        #     maybe handle strings etc but unlikely to need

        is_float = False
        return self._operations[is_float][type(node.op)]

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
            MemRefType(IntegerType(32), [1])
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
        int_32 = IntegerAttr(data, IntegerType(32))
        return [arith.Constant(int_32)]

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
        alloc = memref.Alloc([], [], MemRefType(IndexType(), [1]))
        self.symbol_table[node.target.id] = alloc

        loop_variable = for_loop.body.block.args[0]
        store = memref.Store.get(loop_variable, alloc, [])

        loop_body_ops = [store] + self.generate_body_ops(node) + [scf.Yield()]
        block.add_ops(loop_body_ops)

        return var_allocations + [from_expr, to_expr, start, end, step, alloc, for_loop]

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
        fresh_variables = []

        for statement in node.body:
            fresh_variables += self.get_assigned_variables(statement)

        fresh_variables = remove_duplicates(fresh_variables)

        allocations = []

        for var in list(fresh_variables):
            memory = memref.Alloc([], [], MemRefType(IntegerType(32), [1]))
            self.register_symbol(var, memory)
            allocations.append(memory)

        return allocations


    def register_symbol(self, symbol: str, memory: memref.Alloc):
        self.symbol_table[symbol] = memory

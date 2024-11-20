import ast

from xdsl.dialects import builtin, func, arith, memref
from xdsl.dialects.builtin import IntegerAttr, IntegerType, FunctionType, MemRefType, ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.ir import Operation, Region, Block, SSAValue, OpResult
from typing import Dict, List, Optional, cast, Tuple

from frontend.memref_context import MemrefContext

DoubleOp = Tuple[Operation, Operation]
TripleOp = Tuple[Operation, Operation, Operation]


class PythonToMLIR(ast.NodeVisitor):
    """
    Parses a Python AST to create operations from the tt_data, tt_compute,
    scf, and memref dialects.
    """
    def __init__(self):
        super().__init__()
        self.symbol_table = MemrefContext()  # variable names -> memref
        self.ssa_to_op: Dict[SSAValue, Operation] = {}  # SSA ref -> creator op

        self.operations: List[Operation] | ModuleOp = []

    def generic_visit(self, node):
        print("Missing handling for: " + node.__class__.__name__)
        raise Exception(f"Unhandled construct, no parser provided: {node.__class__.__name__}")

    def visit_Module(self, node):
        operations: list[Operation] = []

        # at the 'module' level should just be a single function def
        for child in node.body:
            op = self.visit(child)
            operations.append(op)

        # after all processing is done, wrap in module operation
        self.operations = builtin.ModuleOp(operations)

    def visit_FunctionDef(self, node) -> FuncOp:
        operations: list[Operation] = []

        # set the current scope
        self.operations = operations

        for child in node.body:
            op = self.visit(child)
            if isinstance(op, Operation):
                operations.append(op)
            if isinstance(op, Tuple):
                operations.extend(op)

        block = Block(operations)
        region = Region(block)

        # return some function definition with contents
        return func.FuncOp(
            node.name,
            FunctionType.from_lists([], []),
            region
        )

    def visit_Assign(self, node) -> DoubleOp | TripleOp:
        # visit RHS, e.g. Constant in 'a = 0'
        rhs: Operation = self.visit(node.value)

        # get the variable name to store the result in
        dest = node.targets[0]
        assert isinstance(dest, ast.Name)
        var_name = dest.id

        # TODO: If first time seeing variable name, create memory space
        #     otherwise refer to prev memory space
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
            return rhs, store

        return rhs, location, store

    def visit_Constant(self, node) -> Operation:
        data = node.value
        int_32 = IntegerAttr(data, IntegerType(32))
        return arith.Constant(int_32)

    def visit_BinOp(self, node) -> Operation:
        lhs: Operation = self.visit(node.left)  # operation that created a, same operation returned from visit_Assign
        rhs: Operation = self.visit(node.right)

        if lhs not in self.operations:
            self.operations.append(lhs)

        if rhs not in self.operations:
            self.operations.append(rhs)

        # get references to the SSA values
        l_val: OpResult = lhs.results[0]
        r_val: OpResult = rhs.results[0]

        return arith.Addi(
            l_val,
            r_val,
            None
        )

    def visit_Name(self, node) -> Operation:
        location = self.symbol_table[node.id]
        load = memref.Load.get(location, [])
        return load

    def register_symbol(self, symbol: str, memory: memref.Alloc):
        self.symbol_table[symbol] = memory


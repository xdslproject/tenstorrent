import ast

from xdsl.dialects import builtin, func, arith, memref, scf
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import IntegerAttr, IntegerType, FunctionType, MemRefType, ModuleOp, IndexType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import Store
from xdsl.ir import Operation, Region, Block, SSAValue, OpResult
from typing import Dict, List, Optional, cast, Tuple

from xdsl.irdl import IRDLOperation

from frontend.memref_context import MemrefContext

DoubleOp = Tuple[Operation, Operation]
TripleOp = Tuple[Operation, Operation, Operation]


def get_vars_assigned_to(ops: List[Operation]) -> List[SSAValue]:
    variables = []

    for op in ops:
        if isinstance(op, Store):
            variable = op.operands[0]
            variables.append(variable)

    return variables


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

    def get_operation(self, node) -> type:
        assert isinstance(node, ast.BinOp)
        # TODO: will need to check if either child is floating point (recursively)
        #     maybe handle strings etc but unlikely to need

        is_float = False
        return self._operations[is_float][type(node.op)]


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

    def visit_BinOp(self, node: ast.BinOp) -> Operation:
        lhs: Operation = self.visit(node.left)  # operation that created a, same operation returned from visit_Assign
        rhs: Operation = self.visit(node.right)

        if lhs not in self.operations:
            self.operations.append(lhs)

        if rhs not in self.operations:
            self.operations.append(rhs)

        # get references to the SSA values
        l_val: OpResult = lhs.results[0]
        r_val: OpResult = rhs.results[0]

        op_constructor = self.get_operation(node)
        return op_constructor(
            l_val,
            r_val,
            None
        )

    def visit_Name(self, node: ast.Name) -> Operation:
        location = self.symbol_table[node.id]
        load = memref.Load.get(location, [])
        return load

    def visit_For(self, node: ast.For):
        contents: List[Operation] = []
        for statement in node.body:
            result = self.visit(statement)
            if isinstance(result, tuple):
                contents.extend(result)
            else:
                contents.append(result)

        from_expr = self.visit(node.iter.args[0])
        to_expr = self.visit(node.iter.args[1])

        start = arith.IndexCastOp(from_expr, IndexType())
        end = arith.IndexCastOp(to_expr, IndexType())

        # lb, up, step, iteration arguments, body
        step = arith.Constant.create(
            properties={
                "value": IntegerAttr.from_index_int_value(1)
            },
            result_types=[IndexType()]
        )

        iter_args = get_vars_assigned_to(contents)

        block_arg_types = [IndexType()]
        block_args = []
        for ssa_val in iter_args:
            block_arg_types.append(ssa_val.type)
            block_args.append(ssa_val)

        yield_stmt = scf.Yield(*block_args)

        block = Block(arg_types=block_arg_types)
        block.add_ops(contents + [yield_stmt])

        body = Region()
        body.add_block(block)

        return from_expr, to_expr, start, end, step, scf.For(
            start.results[0],
            end.results[0],
            step.results[0],
            iter_args,
            body
        )

    def register_symbol(self, symbol: str, memory: memref.Alloc):
        self.symbol_table[symbol] = memory


import ast
from typing import Dict
from xdsl.utils.hints import isa
from xdsl.dialects.builtin import IntegerType, Float32Type, IndexType, NoneType, MemRefType

from .dummy import *
from tenstorrent.dialects import *


MLIRType = IntegerType | Float32Type | IndexType | NoneType


def types_equal(a, b) -> bool:
    int_comparable = [IntegerType, IndexType]
    equal = a == b
    return equal or (type(a) in int_comparable and type(b) in int_comparable)


class TypeChecker(ast.NodeVisitor):
    def __init__(self):
        self.types: Dict[str, MLIRType] = {
            cb_push_back.__name__: NoneType(),
            cb_wait_front.__name__: NoneType(),
            cb_pop_front.__name__: NoneType(),
            cb_reserve_back.__name__: NoneType(),
            cb_pages_available_at_front.__name__: IntegerType(1),
            cb_pages_reservable_at_back.__name__: IntegerType(1),
            noc_semaphore_set.__name__: NoneType(),
            noc_semaphore_set_multicast.__name__: NoneType(),
            noc_async_write_multicast.__name__: NoneType(),
            noc_async_write.__name__: NoneType(),
            noc_async_read.__name__: NoneType(),
            noc_semaphore_inc.__name__: NoneType(),
            noc_semaphore_wait.__name__: NoneType(),
            noc_async_read_barrier.__name__: NoneType(),
            noc_async_write_barrier.__name__: NoneType(),
            "CreateDevice": Device,
            "Core": CoreCoord,
            "DRAMConfig": DRAMBufferConfig,
            "CreateBuffer": Buffer,
            "GetCommandQueue": CommandQueue,
            "EnqueueWriteBuffer": None,
            "EnqueueReadBuffer": None,
            "CreateProgram": Program,
            "Kernel": Kernel,
            "SetRuntimeArgs": None,
            "EnqueueProgram": None,
            "Finish": None,
            "CloseDevice": None
        }

    def generic_visit(self, node):
        raise Exception(f"Unhandled node type {node.__class__.__name__}")

    def visit_Import(self, node):
        pass

    def visit_Pass(self, node):
        pass


    def dominating_type(self, a, b) -> MLIRType:
        if a == b:
            return a

        if a == Float32Type() or b == Float32Type():
            return Float32Type()

        if a == IntegerType(32) or b == IntegerType(32):
            return IntegerType(32)

        raise NotImplementedError(f"Type not in type hierarchy: {a.__class__.__name__}")

    def visit_List(self, node: ast.List):
      assert len(node.elts) == 1
      element_type=self.visit(node.elts[0])
      return MemRefType(element_type, [])


    def visit_Constant(self, node: ast.Constant):
        data = node.value

        if isinstance(data, bool):
            return IntegerType(1)

        if isinstance(data, int):
            return IntegerType(32)

        # Wormhole FPU supports up to single precision floating point
        if isinstance(data, float):
            return Float32Type()

        raise Exception(f"Unhandled constant type: {data.__class__.__name__}")

    def visit_Name(self, node: ast.Name):
        return self.types[node.id]

    def visit_Assign(self, node: ast.Assign):
        """
        On assignment be sure to register the type of a variable if it is not
        already registered, if it is then verify the type.
        """
        if isa(node.targets[0], ast.Name):
          target = node.targets[0].id
        elif isa(node.targets[0], ast.Subscript):
          target = node.targets[0].value.id
        else:
          assert False
        expected_type = self.visit(node.value)

        self.types[target] = expected_type if target not in self.types else (
            self.dominating_type(self.types[target], expected_type)
        )


    def visit_UnaryOp(self, node) -> MLIRType:
        return self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> MLIRType:
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)

        if isinstance(node.op, ast.Div):
            return Float32Type()

        if types_equal(left_type, right_type):
            return left_type

        return self.dominating_type(left_type, right_type)

    def visit_Expr(self, node) -> MLIRType:
        return self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> MLIRType:
        if isa(node.func, ast.Attribute):
          name=node.func.attr
        else:
          name = node.func.id
        if name in self.types:
            return self.types[name]

        raise NotImplementedError(f"Unhandled call: {name}")

    # ********* Generic visits *********
    def visit_Module(self, node):
        for child in node.body:
            self.visit(child)

    def visit_FunctionDef(self, node):
        for child in node.body:
            self.visit(child)

    def visit_For(self, node):
        identifier = node.target.id
        t = IntegerType(32)

        if identifier in self.types:
            assert self.types[identifier] == t

        self.types[identifier] = t

        for child in node.body:
            self.visit(child)

    def visit_While(self, node):
        for child in node.body:
            self.visit(child)

    def visit_If(self, node):
        for child in node.body:
            self.visit(child)

    def print_types(self):
        for key in self.types:
            print(f"{key}: {self.types[key].__class__.__name__}")

    def visit_BoolOp(self, node):
        return IntegerType(1)

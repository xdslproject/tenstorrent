import ast

from xdsl.dialects.builtin import IntegerType, Float32Type, IndexType


def types_equal(a, b) -> bool:
    int_comparable = [IntegerType, IndexType]
    equal = a == b
    return equal or (type(a) in int_comparable and type(b) in int_comparable)


class TypeChecker(ast.NodeVisitor):
    def __init__(self):
        self.types = {}  # str (variable name) -> type in MLIR

    def generic_visit(self, node):
        raise Exception(f"Unhandled node type {node.__class__.__name__}")

    def visit_Constant(self, node: ast.Constant):
        data = node.value

        if isinstance(data, int):
            return IntegerType(32)

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
        target = node.targets[0].id
        expected_type = self.visit(node.value)

        if target in self.types:
            assert types_equal(self.types[target], expected_type)

        self.types[target] = expected_type

    def visit_BinOp(self, node: ast.BinOp):
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        assert types_equal(left_type, right_type)
        return left_type

    # ********* Generic visits *********
    def visit_Module(self, node):
        for child in node.body:
            self.visit(child)

    def visit_FunctionDef(self, node):
        for child in node.body:
            self.visit(child)

    def visit_For(self, node):
        identifier = node.target.id
        t = IndexType()

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

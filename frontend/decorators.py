import ast
import inspect
import sys
from functools import wraps

from xdsl.printer import Printer

from backend.print_metal import PrintMetal
from frontend.python_to_mlir import PythonToMLIR
from frontend.type_checker import TypeChecker


def data_in(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        width = 50
        gap = "\n\n"
        border0 = '\n' + '%' * width + '\n'
        border1 = '=' * width + "\nMLIR" + gap
        border2 = "\n\n" + '*' * width + "\nC++" + gap

        source = inspect.getsource(func)
        tree = ast.parse(source)

        print(border0 + "Python\n")
        print(source)

        type_checker = TypeChecker()
        type_checker.visit(tree)
        type_checker.print_types()
        print()

        tree_walker = PythonToMLIR(type_checker)
        tree_walker.visit(tree)

        printer = Printer(stream=sys.stdout)
        printer.print(border1)
        printer.print_op(tree_walker.operations)
        printer.print(border2)

        out_printer = PrintMetal()
        out_printer.print_module(tree_walker.operations)

        return func(*args, **kwargs)

    return wrapper



import ast
import inspect
import sys
from functools import wraps

from xdsl.printer import Printer

from backend.print_metal import PrintMetal
from frontend.python_to_mlir import PythonToMLIR


def data_in(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        source = inspect.getsource(func)
        tree = ast.parse(source)

        tree_walker = PythonToMLIR()
        tree_walker.visit(tree)

        width = 70
        gap = "\n\n"
        border1 = '=' * width + gap
        border2 = "\n\n" + '*' * width + gap

        printer = Printer(stream=sys.stdout)
        printer.print(border1)
        printer.print_op(tree_walker.operations)
        printer.print(border2)

        out_printer = PrintMetal()
        out_printer.print_module(tree_walker.operations)
        print()

        return func(*args, **kwargs)

    return wrapper



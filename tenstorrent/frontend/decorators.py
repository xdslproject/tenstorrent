import ast
import inspect
import sys
from functools import wraps

from xdsl.printer import Printer

from tenstorrent.backend.print_metalium import PrintMetalium
from tenstorrent.frontend.python_to_mlir import PythonToMLIR
from tenstorrent.frontend.type_checker import TypeChecker


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
        print()

        tree_walker = PythonToMLIR(type_checker)
        tree_walker.visit(tree)

        printer = Printer(stream=sys.stdout)
        printer.print(border1)
        printer.print_op(tree_walker.operations)
        printer.print(border2)

        out_printer = PrintMetalium()
        out_printer.print_op(tree_walker.operations)

        # print to file
        file_name = func.__name__

        import os
        prefix = os.getcwd() + "/tests/results/"
        mlir_file_path = prefix + file_name + ".mlir"
        cpp_file_path = prefix + file_name + ".cpp"

        with open(mlir_file_path, "w") as file:
            mlir_printer = Printer(stream=file)
            mlir_printer.print_op(tree_walker.operations)
            mlir_printer.print('\n')

        with open(cpp_file_path, "w") as file:
            cpp_printer = PrintMetalium(file)
            cpp_printer.print_op(tree_walker.operations)

        return func(*args, **kwargs)

    return wrapper


compute = data_in


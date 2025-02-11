import sys
from xdsl.printer import Printer

from tenstorrent.frontend.python_to_mlir import PythonToMLIR
from tenstorrent.frontend.type_checker import TypeChecker
from tenstorrent.frontend.dummy import *


def _empty_wrapper(*args, **kwargs):
    def empty(*ar):
        pass

    def compile_wrapper(func):
        return empty

    return compile_wrapper


# API Function decorators
data_in = _empty_wrapper
"""
For compiling to a Metalium data-movement kernel (receiving data in)
"""

data_out = _empty_wrapper
"""
For compiling to a Metalium data-movement kernel (sending data out)
"""

compute = _empty_wrapper
"""
For compiling to a Metalium compute-core kernel
"""


host = _empty_wrapper
"""
For functions to be ran as the Metalium host device
"""


def entry_point():
    """
    To be run on import
    """
    code_contents_handle = open(sys.argv[0], "r")
    code_contents = code_contents_handle.read()

    parsed_ast = ast.parse(code_contents)
    code_contents_handle.close()

    type_checker = TypeChecker()
    type_checker.visit(parsed_ast)

    tree_walker = PythonToMLIR(type_checker)
    tree_walker.visit(parsed_ast)

    printer = Printer(stream=sys.stdout)
    printer.print_op(tree_walker.operations)


# This will run whenever this module is imported into a Python script
entry_point()
exit(0)

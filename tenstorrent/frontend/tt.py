import ast, inspect
import sys
from xdsl.printer import Printer

from tenstorrent.frontend.python_to_mlir import PythonToMLIR
from tenstorrent.frontend.type_checker import TypeChecker

def data_in(*args, **kwargs):
  def empty(*args):
    pass

  def compile_wrapper(func):
    return empty

  return compile_wrapper

def entryPoint():
  code_contents_handle = open(sys.argv[0],'r')
  code_contents=code_contents_handle.read()

  parsedAST = ast.parse(code_contents)
  code_contents_handle.close()

  type_checker = TypeChecker()
  type_checker.visit(parsedAST)

  tree_walker = PythonToMLIR(type_checker)
  tree_walker.visit(parsedAST)

  printer = Printer(stream=sys.stdout)
  printer.print_op(tree_walker.operations)

entryPoint()
exit(0)

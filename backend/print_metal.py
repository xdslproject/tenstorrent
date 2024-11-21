from typing import Optional

from xdsl.dialects.builtin import ModuleOp, Operation
from xdsl.dialects.func import FuncOp
from xdsl.dialects.arith import Constant, Addi, Muli, SignlessIntegerBinaryOperation, IndexCastOp
from xdsl.dialects.scf import For, Yield
from xdsl.dialects.memref import Alloc, Store, Load
from xdsl.ir import Block, Region, SSAValue, OpResult


class PrintMetal:
    """
    Prints the Tenstorrent Metal API (C) given a list of xDSL operations
    """
    def __init__(self):
        self._indent = 0
        self._names = {}  # Load -> Variable Name
        self._op_to_sym = {
            Addi: "+",
            Muli: "*"
        }

        self._skip = [Constant, Alloc, Load, Addi, Muli, IndexCastOp, Yield]

    def print_block(self, block: Block):
        operation = block.ops.first

        while operation:
            if isinstance(operation, FuncOp):
                self.print_func(operation)
                operation = operation.next_op

            elif isinstance(operation, Alloc):
                self.print_declaration(operation)
                operation = operation.next_op

            elif isinstance(operation, Store):
                self.print_assignment(operation)
                operation = operation.next_op

            elif isinstance(operation, For):
                self.print_for_loop(operation)
                operation = operation.next_op

            # skip constants on their own, will be picked up later if used
            elif type(operation) in self._skip:
                operation = operation.next_op
                continue

            else:
                self.print(f"UNHANDLED OPERATION: {operation.__class__.__name__}")
                break


    def print_module(self, module: ModuleOp):
        for region in module.regions:
            for block in region.blocks:
                self.print_block(block)

    def print_func(self, func: FuncOp):
        """
        void func_name(typea a, typeb b, ...) {

        }
        """
        self.print(f"void {func.sym_name.data}() {'{'}")

        self._indent += 1
        self.print_body(func)
        self._indent -= 1

        self.print("}")

    def print_for_loop(self, loop: For):
        i = self.create_fresh_variable('i')
        self.print(f"int {i};")
        self.print(f"for ({i} = 0; {i} < 5; {i}++) {'{'}")

        self._indent += 1
        self.print_body(loop)
        self._indent -= 1

        self.print("}")

    def print_body(self, parent: FuncOp | For):
        body: Region = parent.body

        for block in body.blocks:
            self.print_block(block)


    def print_declaration(self, op: Alloc):
        var_name = self.create_fresh_variable()
        type_decl = "std::int32_t "

        self.print(type_decl + var_name + ';')

        loc = op.results[0]
        self._names[loc] = var_name


    def print_assignment(self, op: Store):
        loc = op.operands[1]

        # find what is being written to our location
        ssa_value = op.operands[0]
        result = self.get_value(ssa_value)
        var_name = self._names[loc]

        self.print(f"{var_name} = {result};")


    def create_fresh_variable(self, base='a') -> str:
        names = self._names.values()
        if base not in names:
            return base

        count = 0
        name = base
        while name in names:
            count += 1
            name = base + str(count)

        return name

    def get_value(self, ssa_value: SSAValue) -> str:
        """
        Returns a textual representation of the expression that the ssa_value
        is assigned to.
        """
        creator = ssa_value.owner
        if isinstance(creator, Constant):
            return str(creator.value.value.data)

        if isinstance(creator, SignlessIntegerBinaryOperation):
            return self.binary_op_string(creator)

        raise Exception(f"Unhandled type {creator.__class__} in get_value()")


    def print(self, s: str, indented: bool = True):
        prefix = self._prefix if indented else ""
        print(prefix + s)


    def binary_op_string(self, operation: SignlessIntegerBinaryOperation):
        """
        In a binary operation, each operand will either be a constant, load, or
        another binary operation. This method handles each case and produces a
        string.
        """
        op_str = self._op_to_sym[type(operation)]
        values = ["ERROR", "ERROR"]

        for i in range(0, 2):
            operand = operation.operands[i]
            assert isinstance(operand, OpResult)
            creator = operand.op

            if isinstance(creator, Constant):
                values[i] = creator.value.value.data
            elif isinstance(creator, Load):
                values[i] = self._names[creator.operands[0]]
            elif isinstance(creator, SignlessIntegerBinaryOperation):
                values[i] = '(' + self.binary_op_string(creator) + ')'
            else:
                raise Exception(f"Unhandled type: {operation.__class__} in binary_op_string")

        return f"{values[0]} {op_str} {values[1]}"

    @property
    def _prefix(self):
        return " " * 4 * self._indent

from xdsl.dialects.builtin import ModuleOp, IndexType, IntegerType, Float32Type, IntegerAttr
from xdsl.dialects.func import FuncOp
from xdsl.dialects.arith import Constant, Addi, Muli, Addf, Mulf, SignlessIntegerBinaryOperation, IndexCastOp, \
    FloatingPointLikeBinaryOperation, Cmpi, AndI, OrI, Cmpf, ComparisonOperation, XOrI, Subi, Subf, ExtFOp
from xdsl.dialects.scf import For, Yield, If, While
from xdsl.dialects.memref import Alloc, Store, Load
from xdsl.ir import Block, Region, SSAValue, OpResult


ArithmeticOperation = SignlessIntegerBinaryOperation | FloatingPointLikeBinaryOperation
BooleanOperation = AndI | OrI | Cmpi | Cmpf
BinaryOperation = ArithmeticOperation | BooleanOperation
OpWithBody = FuncOp | For | While

TRUE = IntegerAttr.from_int_and_width(1, 1)


class PrintMetalium:
    """
    Prints the Tenstorrent Metalium API (C) given a list of xDSL operations
    """
    def __init__(self, file=None):
        self._indent = 0
        self._file = file
        self._names = {}  # SSAVal -> Variable Name
        self._op_to_sym = {
            Addi: "+",
            Muli: "*",
            Addf: "+",
            Mulf: "*",
            AndI: "&&",
            OrI: "||",
            XOrI: "^",
            Subi: "-",
            Subf: "-",
        }

        self._int_comparison_ops = {
            IntegerAttr.from_int_and_width(0, 64): "==",
            IntegerAttr.from_int_and_width(1, 64): "!=",

            # signed
            IntegerAttr.from_int_and_width(2, 64): "<",
            IntegerAttr.from_int_and_width(3, 64): "<=",
            IntegerAttr.from_int_and_width(4, 64): ">",
            IntegerAttr.from_int_and_width(5, 64): ">=",

            # unsigned
            IntegerAttr.from_int_and_width(6, 64): "<",
            IntegerAttr.from_int_and_width(7, 64): "<=",
            IntegerAttr.from_int_and_width(8, 64): ">",
            IntegerAttr.from_int_and_width(9, 64): ">=",
        }

        self._float_comparison_ops = {
            # ordered
            IntegerAttr.from_int_and_width(1, 64): "==",
            IntegerAttr.from_int_and_width(2, 64): ">",
            IntegerAttr.from_int_and_width(3, 64): ">=",
            IntegerAttr.from_int_and_width(4, 64): "<",
            IntegerAttr.from_int_and_width(5, 64): "<=",
            IntegerAttr.from_int_and_width(6, 64): "!=",
        }

        self._mlir_to_cpp_type = {
            IndexType(): "std::int32_t",
            IntegerType(32): "std::int32_t",
            Float32Type(): "float",
        }

        self._skip = [
            Constant, Alloc, Load, Addi, Muli, Addf, Mulf, IndexCastOp, Yield,
            Cmpi, AndI, OrI, XOrI, Subi, Subf, ExtFOp
        ]

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

            elif isinstance(operation, If):
                self.print_if_statement(operation)
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
        self.print_region(func.body)
        self._indent -= 1

        self.print("}")

    def print_for_loop(self, loop: For):
        # we know the first operation in the loop should be the store into i
        store_i = loop.body.block.first_op
        i_loop_ssa = store_i.operands[0]
        i_register = store_i.operands[1]

        i = self._names[i_register]
        self._names[i_loop_ssa] = i

        start = self.get_value(loop.lb)
        stop = self.get_value(loop.ub)
        step = self.get_value(loop.step)
        self.print(f"for ({i} = {start}; {i} < {stop}; {i} += {step}) {'{'}")

        self._indent += 1
        self.print_region(loop.body)
        self._indent -= 1

        self.print("}")

    def print_region(self, body: Region):
        for block in body.blocks:
            self.print_block(block)


    def print_declaration(self, op: Alloc):
        index = isinstance(op.next_op, For)
        var_name = self.create_fresh_variable(hint='i' if index else 'a')
        type_decl = self._mlir_to_cpp_type[op.result_types[0].element_type]

        self.print(type_decl + " " + var_name + ";")

        ssa_referring_to_var = op.results[0]
        self._names[ssa_referring_to_var] = var_name


    def print_assignment(self, op: Store):
        # memref.store value, destination[]
        ssa_value = op.operands[0]
        ssa_destination = op.operands[1]

        result = self.get_value(ssa_value)
        var_name = self._names[ssa_destination]

        self.print(f"{var_name} = {result};")


    def print_if_statement(self, op: If):
        self.print(f"if ({self.get_value(op.cond)}) {'{'}")

        self._indent += 1
        self.print_block(op.true_region.blocks[0])
        self._indent -= 1

        or_else = len(op.false_region.blocks) > 0

        self.print("}" + (" else {" if or_else else ""))

        # here need to print the or-else
        if or_else:
            self._indent += 1
            self.print_region(op.false_region)
            self._indent -= 1
            self.print("}")


    def create_fresh_variable(self, hint='a') -> str:
        names = self._names.values()
        if hint not in names:
            return hint

        count = 0
        name = hint
        while name in names:
            count += 1
            name = hint + str(count)

        return name

    def get_value(self, ssa_value: SSAValue) -> str:
        """
        Returns a textual representation of the expression that the ssa_value
        is assigned to.
        """
        if ssa_value in self._names:
            return self._names[ssa_value]

        creator = ssa_value.owner
        if isinstance(creator, Constant):
            return str(creator.value.value.data).lower()

        if isinstance(creator, IndexCastOp):
            return self.get_value(creator.operands[0])

        if isinstance(creator, ExtFOp):
            return "static_cast<float>(" + self.get_value(creator.operands[0]) + ")"

        if isinstance(creator, BinaryOperation):
            return self.binary_op_string(creator)

        if isinstance(creator, Load):
            return self.get_value(creator.operands[0])

        raise Exception(f"Unhandled type {creator.__class__} in get_value()")


    def print(self, s: str, indented: bool = True):
        prefix = self._prefix if indented else ""
        if self._file:
            print(prefix + s, file=self._file)
        else:
            print(prefix + s)


    def binary_op_string(self, operation: BinaryOperation):
        """
        In a binary operation, each operand will either be a constant, load, or
        another binary operation. This method handles each case and produces a
        string.
        """
        if isinstance(operation, ComparisonOperation):
            # TODO: add support for floating point comparison
            op_str = self._int_comparison_ops[operation.predicate]

        # whilst XOrI is a binary operation, we know it can encode 'not'
        elif isinstance(operation, XOrI) and operation.operands[1].op.value == TRUE:
            return "!(" + self.binary_op_string(operation.operands[0].op) + ")"

        else:
            op_str = self._op_to_sym[type(operation)]

        values = ["ERROR", "ERROR"]

        for i in range(0, 2):
            operand = operation.operands[i]
            assert isinstance(operand, OpResult)
            creator = operand.op

            if isinstance(creator, Constant):
                values[i] = creator.value.value.data

            elif isinstance(creator, ExtFOp):
                values[i] = "static_cast<float>(" + self.get_value(creator.operands[0]) + ")"

            elif isinstance(creator, Load):
                values[i] = self._names[creator.operands[0]]

            elif isinstance(creator, ArithmeticOperation):
                values[i] = '(' + self.binary_op_string(creator) + ')'

            elif isinstance(creator, BooleanOperation):
                values[i] = self.binary_op_string(creator)
            else:
                raise Exception(f"Unhandled type: {operation.__class__} in binary_op_string")

        return f"{values[0]} {op_str} {values[1]}"

    @property
    def _prefix(self):
        return " " * 4 * self._indent

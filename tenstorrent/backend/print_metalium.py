from xdsl.dialects.builtin import ModuleOp, IndexType, Float32Type, IntegerAttr, i1, i32, f32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.arith import ConstantOp, AddiOp, MuliOp, AddfOp, MulfOp, SignlessIntegerBinaryOperation, IndexCastOp, \
    FloatingPointLikeBinaryOperation, CmpiOp, AndIOp, OrIOp, CmpfOp, ComparisonOperation, XOrIOp, SubiOp, SubfOp, ExtFOp, DivfOp
from xdsl.dialects.scf import ForOp, YieldOp, IfOp, WhileOp
from xdsl.dialects.memref import AllocOp, StoreOp, LoadOp
from xdsl.ir import Block, Region, OpResult, Attribute

from tenstorrent.dialects import *


ArithmeticOperation = SignlessIntegerBinaryOperation | FloatingPointLikeBinaryOperation
BooleanOperation = AndIOp | OrIOp | CmpiOp | CmpfOp
BinaryOperation = ArithmeticOperation | BooleanOperation
OpWithBody = FuncOp | ForOp | WhileOp
CircularBufferOperationWithResult = CBPagesAvailableAtFront | CBPagesReservableAtBack

TRUE = IntegerAttr.from_int_and_width(1, 1)
TenstorrentOps = list(DataMovement.operations) + list(Compute.operations) + list(TTHost.operations) + list(CircularBuffer.operations)

# TODO: currently printing API calls/func names is hardcoded, but should not vary based on whether the function is
#     used on its own (as a statement) or used on the rhs (as an expression). Should handle this to remove a lot
#     of the boilerplate.


class PrintMetalium:
    """
    Prints the Tenstorrent Metalium API (C) given a list of xDSL operations
    """
    def __init__(self, file=None):
        self._indent = 0
        self._file = file
        self._names = {}  # SSAVal -> Variable Name
        self._op_to_sym = {
            AddiOp: "+",
            MuliOp: "*",
            AddfOp: "+",
            MulfOp: "*",
            AndIOp: "&&",
            OrIOp: "||",
            XOrIOp: "^",
            SubiOp: "-",
            SubfOp: "-",
            DivfOp: "/",
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
            i32: "std::int32_t",
            f32: "float",
            i1: "bool",
        }

        self._skip = [
            ConstantOp, LoadOp, AddiOp, MuliOp, AddfOp, MulfOp, IndexCastOp, YieldOp,
            CmpiOp, AndIOp, OrIOp, XOrIOp, SubiOp, SubfOp, ExtFOp, DivfOp,
            CBPagesReservableAtBack, CBPagesAvailableAtFront, ReturnOp
        ]

    def print_op(self, operation):
        if isinstance(operation, ModuleOp):
            for region in operation.regions:
                for block in region.blocks:
                    self.print_op(block)

        # skip constants on their own, will be picked up later if used
        elif type(operation) in self._skip:
            pass

        elif isinstance(operation, Block):
            for op in operation.ops:
                self.print_op(op)

        elif isinstance(operation, FuncOp):
            self.print_func_def(operation)

        elif isinstance(operation, AllocOp):
            self.print_declaration(operation)

        elif isinstance(operation, StoreOp):
            self.print_assignment(operation)

        elif isinstance(operation, ForOp):
            self.print_for_loop(operation)

        elif isinstance(operation, IfOp):
            self.print_if_statement(operation)

        elif type(operation) in TenstorrentOps:
            self.print_tt_op(operation)

        else:
            raise NotImplementedError(f"Unhandled operation: {operation.__class__.__name__}")


    def print_func_def(self, func: FuncOp):
        """
        void func_name(typea a, typeb b, ...) {

        }
        """
        self.print(f"void {func.sym_name.data}() {'{'}")

        self._indent += 1
        self.print_region(func.body)
        self._indent -= 1

        self.print("}")

    def print_for_loop(self, loop: ForOp):
        # we know the first operation in the loop should be the store into i
        store_i = loop.body.block.first_op
        i_loop_ssa = store_i.operands[0]
        i_register = store_i.operands[1]

        i = self._names[i_register]
        self._names[i_loop_ssa] = i

        start = self.get_rhs_value(loop.lb)
        stop = self.get_rhs_value(loop.ub)
        step = self.get_rhs_value(loop.step)
        self.print(f"for ({i} = {start}; {i} < {stop}; {i} += {step}) {'{'}")

        self._indent += 1
        self.print_region(loop.body)
        self._indent -= 1

        self.print("}")

    def print_region(self, body: Region):
        for block in body.blocks:
            self.print_op(block)


    def print_declaration(self, op: AllocOp):
        index = isinstance(op.next_op, ForOp)
        var_name = self.create_fresh_variable(hint='i' if index else 'a')
        type_decl = self._mlir_to_cpp_type[op.result_types[0].element_type]

        self.print(type_decl + " " + var_name + ";")

        ssa_referring_to_var = op.results[0]
        self._names[ssa_referring_to_var] = var_name


    def print_assignment(self, op: StoreOp):
        # memref.store value, destination[]
        ssa_value = op.operands[0]
        ssa_destination = op.operands[1]

        result = self.get_rhs_value(ssa_value)
        var_name = self._names[ssa_destination]

        self.print(f"{var_name} = {result};")


    def print_if_statement(self, op: IfOp):
        self.print(f"if ({self.get_rhs_value(op.cond)}) {'{'}")

        self._indent += 1
        self.print_op(op.true_region.blocks[0])
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

    def get_rhs_value(self, elem: SSAValue | Attribute) -> str:
        """
        Returns a textual representation of the expression that the ssa_value
        is assigned to.
        """
        if elem in self._names:
            return self._names[elem]

        boolean = elem.type.name == 'integer_type' and elem.type.bitwidth == 1
          
        if isinstance(elem, Attribute):
            if boolean and elem.value.data == -1:
                return "true"
            return str(elem.value.data).lower()

        creator = elem.owner
        if isinstance(creator, ConstantOp):
            if boolean and creator.value.value.data == -1:
                return "true"
            return str(creator.value.value.data).lower()            

        if isinstance(creator, IndexCastOp):
            return self.get_rhs_value(creator.operands[0])

        if isinstance(creator, ExtFOp):
            return "static_cast<float>(" + self.get_rhs_value(creator.operands[0]) + ")"

        if isinstance(creator, BinaryOperation):
            return self.binary_op_string(creator)

        if isinstance(creator, LoadOp):
            return self.get_rhs_value(creator.operands[0])

        if isinstance(creator, CircularBufferOperationWithResult):
            arg1 = self.get_rhs_value(creator.operands[0])
            arg2 = self.get_rhs_value(creator.operands[1])
            return f"{creator.name.replace('.', '_')}({arg1}, {arg2})"

        raise Exception(f"Unhandled type {creator.__class__} in get_value()")

    def print_tt_op(self, operation):
        first_two_chars = operation.name[:2]
        if first_two_chars == 'cb':
            api_name = operation.name.replace('.', '_')
        elif first_two_chars == 'dm':
            api_name = operation.name.replace('dm.', '')
        elif first_two_chars == 'co':
            api_name = operation.name.replace('comp.', '')
        else:
            raise ValueError(f"Unsupported operation name: {operation.name}")

        values = [self.get_rhs_value(op) for op in operation.operands]
        template_args = self.template_args_as_string(operation)
        self.print(f"{api_name}{template_args}({', '.join(values)});")


    def template_args_as_string(self, operation: IRDLOperation):
        if not operation.properties:
            return ""

        return f"<{', '.join([self.get_rhs_value(p) for p in operation.properties.values()])}>"

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
        elif isinstance(operation, XOrIOp) and operation.operands[1].op.value == TRUE:
            return "!(" + self.binary_op_string(operation.operands[0].op) + ")"

        else:
            op_str = self._op_to_sym[type(operation)]

        values = ["ERROR", "ERROR"]

        for i in range(0, 2):
            operand = operation.operands[i]
            assert isinstance(operand, OpResult)
            creator = operand.op

            if isinstance(creator, ConstantOp):
                values[i] = creator.value.value.data

            elif isinstance(creator, ExtFOp):
                values[i] = "static_cast<float>(" + self.get_rhs_value(creator.operands[0]) + ")"

            elif isinstance(creator, LoadOp):
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

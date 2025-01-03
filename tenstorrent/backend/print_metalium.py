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
TenstorrentOps = (list(DataMovement.operations)
                  + list(Compute.operations)
                  + list(TTHost.operations)
                  + list(CircularBuffer.operations))


def get_api_name(op_name: str) -> str:
    first_two_chars = op_name[:2]
    match first_two_chars:
        case 'cb':
            return op_name.replace('.', '_')
        case 'dm':
            return op_name.replace('dm.', '')
        case 'co':
            return op_name.replace('comp.', '')
        case default:
            raise Exception(f"Unhandled operation name: {op_name}")


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

        self._skip_next_op = False

    def print_op(self, operation):
        if self._skip_next_op:
            self._skip_next_op = False
            return

        if isinstance(operation, ModuleOp):
            for region in operation.regions:
                for block in region.blocks:
                    self.print_op(block)

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
        var_name = self.create_fresh_variable(hint='i' if index else op.results[0].name_hint)
        type_decl = self._mlir_to_cpp_type[op.result_types[0].element_type]

        nxt = op.next_op
        rhs = ""

        # here we are both a decl and init
        if isinstance(nxt, StoreOp) and nxt.operands[1] == op.results[0]:
            rhs_val = nxt.operands[0]
            rhs = f" = {self.get_rhs_value(rhs_val)}"
            self._skip_next_op = True

        self.print(type_decl + " " + var_name + rhs + ";")
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

    def get_rhs_value(self, elem: SSAValue | Attribute, brackets=False) -> str:
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
            if brackets:
                return f"({self.binary_op_string(creator)})"
            return self.binary_op_string(creator)

        if isinstance(creator, LoadOp):
            return self.get_rhs_value(creator.operands[0])

        if isinstance(creator, CircularBufferOperationWithResult):
            arg1 = self.get_rhs_value(creator.operands[0])
            arg2 = self.get_rhs_value(creator.operands[1])
            return f"{creator.name.replace('.', '_')}({arg1}, {arg2})"

        raise Exception(f"Unhandled type {creator.__class__} in get_value()")

    def print_tt_op(self, operation):
        api_name = get_api_name(operation.name)
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
        # TODO: add support for floating point comparison
        if isinstance(operation, ComparisonOperation):
            op_str = self._int_comparison_ops[operation.predicate]

        # whilst XOrI is a binary operation, we know it can encode 'not'
        elif isinstance(operation, XOrIOp) and operation.operands[1].op.value == TRUE:
            return "!(" + self.get_rhs_value(operation.operands[0]) + ")"

        else:
            op_str = self._op_to_sym[type(operation)]

        brackets = True
        lhs = self.get_rhs_value(operation.operands[0], brackets)
        rhs = self.get_rhs_value(operation.operands[1], brackets)

        return f"{lhs} {op_str} {rhs}"

    @property
    def _prefix(self):
        return " " * 4 * self._indent

import ast
from typing import Dict
from xdsl.utils.hints import isa
from xdsl.ir.core import Attribute
from xdsl.dialects.builtin import (
    IntegerType,
    Float32Type,
    IndexType,
    NoneType,
    MemRefType,
    i32,
    i64,
    i1,
    ContainerType,
)

from .dummy import *
from tenstorrent.dialects import *
from tenstorrent.dialects.ttshared import ConstExprType

TYPE_STR_TO_MLIR_TYPE = {
    "int": IntegerType(32),
    "uint16": IntegerType(16),
}


def types_equal(a, b) -> bool:
    int_comparable = [IntegerType, IndexType]
    equal = a == b
    return equal or (type(a) in int_comparable and type(b) in int_comparable)


class TypeChecker(ast.NodeVisitor):
    def __init__(self):
        self.types: Dict[str, Attribute] = {
            cb_push_back.__name__: NoneType(),
            cb_wait_front.__name__: NoneType(),
            cb_pop_front.__name__: NoneType(),
            cb_reserve_back.__name__: NoneType(),
            cb_pages_available_at_front.__name__: IntegerType(1),
            cb_pages_reservable_at_back.__name__: IntegerType(1),
            noc_semaphore_set.__name__: NoneType(),
            noc_semaphore_set_multicast.__name__: NoneType(),
            noc_async_write_multicast.__name__: NoneType(),
            noc_async_write.__name__: NoneType(),
            noc_async_read.__name__: NoneType(),
            noc_semaphore_inc.__name__: NoneType(),
            noc_semaphore_wait.__name__: NoneType(),
            noc_async_read_barrier.__name__: NoneType(),
            noc_async_write_barrier.__name__: NoneType(),
            get_noc_addr_from_bank_id.__name__: i64,
            get_noc_address.__name__: i64,
            copy.__name__: NoneType(),
            copy_to_dst_init_short_with_dt.__name__: NoneType(),
            copy_to_dst_init_short.__name__: NoneType(),
            copy_init.__name__: NoneType(),
            acquire_dst.__name__: NoneType(),
            release_dst.__name__: NoneType(),
            regs_acquire.__name__: NoneType(),
            regs_wait.__name__: NoneType(),
            regs_commit.__name__: NoneType(),
            regs_release.__name__: NoneType(),
            binary_op_init_common.__name__: NoneType(),
            abs_init.__name__: NoneType(),
            abs.__name__: NoneType(),
            add_init_nof.__name__: NoneType(),
            add_init.__name__: NoneType(),
            add.__name__: NoneType(),
            sub_init_nof.__name__: NoneType(),
            sub_init.__name__: NoneType(),
            sub.__name__: NoneType(),
            mul_init_f.__name__: NoneType(),
            mul_init.__name__: NoneType(),
            mul.__name__: NoneType(),
            add_bcast_cols_init_short.__name__: NoneType(),
            add_bcast_rows_init_short.__name__: NoneType(),
            add_bcast.__name__: NoneType(),
            sub_bcast_cols_init_short.__name__: NoneType(),
            sub_bcast.__name__: NoneType(),
            mul_bcast_cols_init_short.__name__: NoneType(),
            mul_bcast_rows_init_short.__name__: NoneType(),
            mul_bcast.__name__: NoneType(),
            mul_bcast_scalar_init_short.__name__: NoneType(),
            mul_bcast_scalar.__name__: NoneType(),
            mm_init.__name__: NoneType(),
            mm_init_short_with_dt.__name__: NoneType(),
            mm_init_short.__name__: NoneType(),
            matmul.__name__: NoneType(),
            mm_block_init.__name__: NoneType(),
            mm_block_init_short.__name__: NoneType(),
            mm_block_init_short_with_dt.__name__: NoneType(),
            matmul_block.__name__: NoneType(),
            exp_init.__name__: NoneType(),
            exp.__name__: NoneType(),
            exp2init.__name__: NoneType(),
            exp2.__name__: NoneType(),
            exp_m1init.__name__: NoneType(),
            exp_m1.__name__: NoneType(),
            relu_init.__name__: NoneType(),
            relu.__name__: NoneType(),
            relu_max_init.__name__: NoneType(),
            relu_max.__name__: NoneType(),
            relu_min_init.__name__: NoneType(),
            relu_min.__name__: NoneType(),
            leaky_relu_init.__name__: NoneType(),
            elu_init.__name__: NoneType(),
            elu.__name__: NoneType(),
            erf_init.__name__: NoneType(),
            erf.__name__: NoneType(),
            erfc_init.__name__: NoneType(),
            erfc.__name__: NoneType(),
            erfinv_init.__name__: NoneType(),
            erfinv.__name__: NoneType(),
            gelu_init.__name__: NoneType(),
            gelu.__name__: NoneType(),
            heaviside_init.__name__: NoneType(),
            heaviside.__name__: NoneType(),
            is_inf_init.__name__: NoneType(),
            is_inf.__name__: NoneType(),
            is_posinf_init.__name__: NoneType(),
            is_posinf.__name__: NoneType(),
            is_neginf_init.__name__: NoneType(),
            is_neginf.__name__: NoneType(),
            is_finite_init.__name__: NoneType(),
            is_finite.__name__: NoneType(),
            is_na_n.__name__: NoneType(),
            i0init.__name__: NoneType(),
            i0.__name__: NoneType(),
            logical_not_unary_init.__name__: NoneType(),
            logical_not_unary.__name__: NoneType(),
            recip_init.__name__: NoneType(),
            recip.__name__: NoneType(),
            sign_init.__name__: NoneType(),
            sign.__name__: NoneType(),
            sqrt_init.__name__: NoneType(),
            sqrt.__name__: NoneType(),
            r_sqrt_init.__name__: NoneType(),
            r_sqrt.__name__: NoneType(),
            sigmoid_init.__name__: NoneType(),
            sigmoid.__name__: NoneType(),
            log_init.__name__: NoneType(),
            log.__name__: NoneType(),
            log_with_base_init.__name__: NoneType(),
            log_with_base.__name__: NoneType(),
            power_init.__name__: NoneType(),
            power.__name__: NoneType(),
            r_sub_init.__name__: NoneType(),
            r_sub.__name__: NoneType(),
            sign_bit_init.__name__: NoneType(),
            sign_bit.__name__: NoneType(),
            square_init.__name__: NoneType(),
            square.__name__: NoneType(),
            reduce.__name__: NoneType(),
            transpose_wh_init.__name__: NoneType(),
            transpose_wh.__name__: NoneType(),
            tanh_init.__name__: NoneType(),
            tanh.__name__: NoneType(),
            tan_init.__name__: NoneType(),
            tan.__name__: NoneType(),
            sin_init.__name__: NoneType(),
            sin.__name__: NoneType(),
            cos_init.__name__: NoneType(),
            cos.__name__: NoneType(),
            asin_init.__name__: NoneType(),
            asin.__name__: NoneType(),
            atan_init.__name__: NoneType(),
            atan.__name__: NoneType(),
            acos_init.__name__: NoneType(),
            acos.__name__: NoneType(),
            ltz_init.__name__: NoneType(),
            ltz.__name__: NoneType(),
            eqz_init.__name__: NoneType(),
            eqz.__name__: NoneType(),
            lez_init.__name__: NoneType(),
            lez.__name__: NoneType(),
            gtz_init.__name__: NoneType(),
            gtz.__name__: NoneType(),
            gez_init.__name__: NoneType(),
            gez.__name__: NoneType(),
            nez_init.__name__: NoneType(),
            nez.__name__: NoneType(),
            unary_ne_init.__name__: NoneType(),
            unary_ne.__name__: NoneType(),
            unary_gt_init.__name__: NoneType(),
            unary_gt.__name__: NoneType(),
            unary_lt_init.__name__: NoneType(),
            unary_lt.__name__: NoneType(),
            tilize_init.__name__: NoneType(),
            tilize_init_short.__name__: NoneType(),
            tilize_init_short_with_dt.__name__: NoneType(),
            tilize_block.__name__: NoneType(),
            tilize_uninit.__name__: NoneType(),
            tilize_uninit_with_dt.__name__: NoneType(),
            untilize_init.__name__: NoneType(),
            untilize_init_short.__name__: NoneType(),
            untilize_block.__name__: NoneType(),
            untilize_uninit.__name__: NoneType(),
            pack_tile.__name__: NoneType(),
            "CreateDevice": host.Device(),
            "Core": host.CoreCoord(),
            "DRAMConfig": host.DRAMBufferConfig(),
            "CreateBuffer": host.Buffer(),
            "GetCommandQueue": host.CommandQueue(),
            "EnqueueWriteBuffer": None,
            "EnqueueReadBuffer": None,
            "CreateProgram": host.Program(),
            "Kernel": host.Kernel(),
            "SetRuntimeArgs": None,
            "EnqueueProgram": None,
            "Finish": None,
            "CloseDevice": None,
            "GetMemoryAddress": IndexType(),
            "CBConfig": host.CircularBufferConfig(),
            "CreateCircularBuffer": host.CBHandle(),
            "cb_get_write_ptr": i32,
            "cb_get_read_ptr": i32,
            get_compile_time_arg_val.__name__: ConstExprType(i32),
            InterleavedAddrGen.__name__: None,
        }

    def generic_visit(self, node):
        raise Exception(f"Unhandled node type {node.__class__.__name__}")

    def visit_Import(self, node):
        pass

    def visit_ImportFrom(self, node):
        pass

    def visit_Pass(self, node):
        pass

    # TODO: this method needs rethinking... overall should
    #  let constexpr dominate, let biggest bitwidth dominate,
    #  and further let floating point dominate
    @staticmethod
    def dominating_type(a, b) -> Attribute:
        """
        Returns the dominating type, unwrapping ConstExprType if needed
        """
        a = TypeChecker.unwrap(a)
        b = TypeChecker.unwrap(b)

        if a == b:
            return a

        # TODO: create a strict partial order of types
        fp_type = Float32Type()
        if a == fp_type or b == fp_type:
            return fp_type

        if a == IntegerType(32) or b == IntegerType(32):
            return IntegerType(32)

        raise NotImplementedError(f"Type not in type hierarchy: {a.__class__.__name__}")

    @staticmethod
    def one_is_constexpr(*args) -> bool:
        for arg in args:
            if isinstance(arg, ConstExprType):
                return True

        return False

    @staticmethod
    def unwrap(a) -> Attribute:
        # TODO: xDSL has a method for this right..? Use it!
        if isinstance(a, ContainerType):
            return a.get_element_type()

        return a

    def visit_List(self, node: ast.List):
        assert len(node.elts) == 1
        element_type = self.visit(node.elts[0])
        return MemRefType(element_type, [])

    def visit_Constant(self, node: ast.Constant):
        data = node.value

        if isinstance(data, bool):
            return IntegerType(1)

        if isinstance(data, int):
            return IntegerType(32)

        # Wormhole FPU supports up to single precision floating point
        if isinstance(data, float):
            return Float32Type()

        raise Exception(f"Unhandled constant type: {data.__class__.__name__}")

    def visit_Name(self, node: ast.Name):
        return self.types[node.id]

    def visit_Subscript(self, node: ast.Subscript):
        return self.types[node.value.id].element_type

    def visit_Assign(self, node: ast.Assign):
        """
        On assignment be sure to register the type of a variable if it is not
        already registered, if it is then verify the type.
        """
        if isa(node.targets[0], ast.Name):
            target = node.targets[0].id
            expected_type = self.visit(node.value)

            self.types[target] = (
                expected_type
                if target not in self.types
                else (TypeChecker.dominating_type(self.types[target], expected_type))
            )

        # if we are writing to an array, the array should have already been
        # given a type as would have been declared above
        elif isa(node.targets[0], ast.Subscript):
            return
        else:
            assert False

    def visit_UnaryOp(self, node) -> Attribute:
        return self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> Attribute:
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        constexpr = TypeChecker.one_is_constexpr(left_type, right_type)
        runtime_type = None

        if (
            isinstance(left_type, MemRefType)
            and isinstance(right_type, IntegerType)
            and isinstance(node.op, ast.Mult)
        ):
            return MemRefType(left_type.element_type, [node.right.value])

        if isinstance(node.op, ast.Div):
            runtime_type = Float32Type()

        else:
            runtime_type = TypeChecker.dominating_type(left_type, right_type)

        # NOTE: this always assumes the other operand is
        # evaluatable at compile-time
        return ConstExprType(runtime_type) if constexpr else runtime_type

    def visit_Expr(self, node) -> Attribute:
        return self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> Attribute:
        if isa(node.func, ast.Attribute):
            name = node.func.attr
        else:
            name = node.func.id

        if name == "to_array":
            assert node.args[1].id in TYPE_STR_TO_MLIR_TYPE
            return MemRefType(TYPE_STR_TO_MLIR_TYPE[node.args[1].id], [])
        elif name in self.types:
            return self.types[name]

        if name == "print":
            return NoneType

        raise NotImplementedError(f"Unhandled call: {name}")

    def visit_Compare(self, node: ast.Compare):
        t1 = self.visit(node.left)
        t2 = self.visit(node.comparators[0])

        constexpr = TypeChecker.one_is_constexpr(t1, t2)
        return ConstExprType(i1) if constexpr else i1

    # ********* Generic visits *********
    def visit_Module(self, node):
        for child in node.body:
            self.visit(child)

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            self.types[arg.arg] = i32

        for child in node.body:
            self.visit(child)

    def visit_AugAssign(self, node):
        return

    def visit_For(self, node):
        identifier = node.target.id
        t = IntegerType(32)

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

    def visit_BoolOp(self, node):
        return IntegerType(1)

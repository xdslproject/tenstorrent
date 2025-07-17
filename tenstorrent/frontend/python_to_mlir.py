import ast
from typing import Dict, List, Tuple, Optional

from xdsl.dialects import builtin, func, arith, memref, scf, printf
from xdsl.dialects.builtin import (
    FunctionType,
    ModuleOp,
    IndexType,
    FloatAttr,
    Float32Type,
    BoolAttr,
)
from xdsl.ir import Region, Block
from xdsl.irdl import AnyOf, EqAttrConstraint
from xdsl.utils.hints import isa

from tenstorrent.dialects import *
from tenstorrent.utils import flatten, remove_duplicates, subtract
from .dummy import *
from .type_casting import get_cast, cast_if_needed

from enum import Enum


class KernelType(Enum):
    DATA_IN = 1
    COMPUTE = 2
    DATA_OUT = 3
    HOST = 4


NodeWithBody = ast.If | ast.For | ast.While | ast.FunctionDef


TYPE_STR_TO_MLIR_TYPE = {
    "int": builtin.i32,
    "uint": builtin.i32,
    "long": builtin.i64,
    "bool": builtin.i1,
    "half": builtin.f16,
    "float": builtin.f32,
    "double": builtin.f64,
}


class PythonToMLIR(ast.NodeVisitor):
    """
    Parses a Python AST to create operations from the tt_data, tt_compute,
    scf, and memref dialects.
    """

    def __init__(self, type_checker):
        super().__init__()
        self.fn_kernel_type = None
        self.symbol_table = {}  # variable names -> memref

        self.operations: List[Operation] | ModuleOp = []
        self.type_checker = type_checker

        self._uint_comparison: Dict[type, str] = {
            ast.Eq: arith.CMPI_COMPARISON_OPERATIONS[0],
            ast.NotEq: arith.CMPI_COMPARISON_OPERATIONS[1],
            ast.Lt: arith.CMPI_COMPARISON_OPERATIONS[6],
            ast.LtE: arith.CMPI_COMPARISON_OPERATIONS[7],
            ast.Gt: arith.CMPI_COMPARISON_OPERATIONS[8],
            ast.GtE: arith.CMPI_COMPARISON_OPERATIONS[9],
        }

        self._sint_comparison: Dict[type, str] = {
            ast.Eq: arith.CMPI_COMPARISON_OPERATIONS[0],
            ast.NotEq: arith.CMPI_COMPARISON_OPERATIONS[1],
            ast.Lt: arith.CMPI_COMPARISON_OPERATIONS[2],
            ast.LtE: arith.CMPI_COMPARISON_OPERATIONS[3],
            ast.Gt: arith.CMPI_COMPARISON_OPERATIONS[4],
            ast.GtE: arith.CMPI_COMPARISON_OPERATIONS[5],
        }

        # use only ordered comparisons (don't allow NaN)
        self._float_comparison: Dict[type, str] = {
            ast.Eq: arith.CMPF_COMPARISON_OPERATIONS[1],
            ast.Gt: arith.CMPF_COMPARISON_OPERATIONS[2],
            ast.GtE: arith.CMPF_COMPARISON_OPERATIONS[3],
            ast.Lt: arith.CMPF_COMPARISON_OPERATIONS[4],
            ast.LtE: arith.CMPF_COMPARISON_OPERATIONS[5],
            ast.NotEq: arith.CMPF_COMPARISON_OPERATIONS[6],
        }

        self._operations: Dict[
            Attribute, Dict[type(ast.operator), type(IRDLOperation)]
        ] = {
            IntegerType(32): {
                ast.Add: arith.AddiOp,
                ast.Mult: arith.MuliOp,
                ast.Sub: arith.SubiOp,
                ast.Div: arith.DivfOp,
            },
            Float32Type(): {
                ast.Add: arith.AddfOp,
                ast.Mult: arith.MulfOp,
                ast.Sub: arith.SubfOp,
                ast.Div: arith.DivfOp,
            },
        }

        self._functions: Dict[str, type] = {
            # shared
            cb_push_back.__name__: CBPushBack,
            cb_reserve_back.__name__: CBReserveBack,
            cb_pop_front.__name__: CBPopFront,
            cb_wait_front.__name__: CBWaitFront,
            cb_pages_reservable_at_back.__name__: CBPagesReservableAtBack,
            cb_pages_available_at_front.__name__: CBPagesAvailableAtFront,
            # TODO: new class for data movement cores
            noc_async_read.__name__: DMNocAsyncRead,
            noc_async_write.__name__: DMNocAsyncWrite,
            noc_semaphore_set.__name__: DMNocSemaphoreSet,
            noc_semaphore_set_multicast.__name__: DMNocSemaphoreSetMulticast,
            noc_async_write_multicast.__name__: DMNocAsyncWriteMulticast,
            noc_semaphore_wait.__name__: DMNocSemaphoreWait,
            noc_semaphore_inc.__name__: DMNocSemaphoreInc,
            noc_async_read_barrier.__name__: DMNocAsyncReadBarrier,
            noc_async_write_barrier.__name__: DMNocAsyncWriteBarrier,
            get_noc_addr_from_bank_id.__name__: DMGetNocAddrFromBankId,
            get_compile_time_arg_val.__name__: GetCompileTimeArgVal,
            # TODO: Should separate into different classes here for compute
            binary_op_init_common.__name__: BinaryOpInitCommon,
            pack_tile.__name__: PackTile,
            copy.__name__: Copy,
            copy_to_dst_init_short_with_dt.__name__: CopyToDSTInitShortWithDT,
            copy_to_dst_init_short.__name__: CopyToDSTInitShort,
            copy_init.__name__: CopyInit,
            acquire_dst.__name__: AcquireDST,
            release_dst.__name__: ReleaseDST,
            regs_acquire.__name__: RegsAcquire,
            regs_wait.__name__: RegsWait,
            regs_commit.__name__: RegsCommit,
            regs_release.__name__: RegsRelease,
            abs_init.__name__: AbsInit,
            abs.__name__: Abs,
            add_init_nof.__name__: AddInitNof,
            add_init.__name__: AddInit,
            add.__name__: Add,
            sub_init_nof.__name__: SubInitNof,
            sub_init.__name__: SubInit,
            sub.__name__: Sub,
            mul_init_f.__name__: MulInitF,
            mul_init.__name__: MulInit,
            mul.__name__: Mul,
            add_bcast_cols_init_short.__name__: AddBcastColsInitShort,
            add_bcast_rows_init_short.__name__: AddBcastRowsInitShort,
            add_bcast.__name__: AddBcast,
            sub_bcast_cols_init_short.__name__: SubBcastColsInitShort,
            sub_bcast.__name__: SubBcast,
            mul_bcast_cols_init_short.__name__: MulBcastColsInitShort,
            mul_bcast_rows_init_short.__name__: MulBcastRowsInitShort,
            mul_bcast.__name__: MulBcast,
            mul_bcast_scalar_init_short.__name__: MulBcastScalarInitShort,
            mul_bcast_scalar.__name__: MulBcastScalar,
            mm_init.__name__: MMInit,
            mm_init_short_with_dt.__name__: MMInitShortWithDT,
            mm_init_short.__name__: MMInitShort,
            matmul.__name__: Matmul,
            mm_block_init.__name__: MMBlockInit,
            mm_block_init_short.__name__: MMBlockInitShort,
            mm_block_init_short_with_dt.__name__: MMBlockInitShortWithDT,
            matmul_block.__name__: MatmulBlock,
            exp_init.__name__: ExpInit,
            exp.__name__: Exp,
            exp2init.__name__: Exp2Init,
            exp2.__name__: Exp2,
            exp_m1init.__name__: ExpM1Init,
            exp_m1.__name__: ExpM1,
            relu_init.__name__: ReluInit,
            relu.__name__: Relu,
            relu_max_init.__name__: ReluMaxInit,
            relu_max.__name__: ReluMax,
            relu_min_init.__name__: ReluMinInit,
            relu_min.__name__: ReluMin,
            leaky_relu_init.__name__: LeakyReluInit,
            elu_init.__name__: EluInit,
            elu.__name__: Elu,
            erf_init.__name__: ErfInit,
            erf.__name__: Erf,
            erfc_init.__name__: ErfcInit,
            erfc.__name__: Erfc,
            erfinv_init.__name__: ErfinvInit,
            erfinv.__name__: Erfinv,
            gelu_init.__name__: GeluInit,
            gelu.__name__: Gelu,
            heaviside_init.__name__: HeavisideInit,
            heaviside.__name__: Heaviside,
            is_inf_init.__name__: IsInfInit,
            is_inf.__name__: IsInf,
            is_posinf_init.__name__: IsPosinfInit,
            is_posinf.__name__: IsPosinf,
            is_neginf_init.__name__: IsNeginfInit,
            is_neginf.__name__: IsNeginf,
            is_finite_init.__name__: IsFiniteInit,
            is_finite.__name__: IsFinite,
            is_na_n.__name__: IsNaN,
            i0init.__name__: I0Init,
            i0.__name__: I0,
            logical_not_unary_init.__name__: LogicalNotUnaryInit,
            logical_not_unary.__name__: LogicalNotUnary,
            recip_init.__name__: RecipInit,
            recip.__name__: Recip,
            sign_init.__name__: SignInit,
            sign.__name__: Sign,
            sqrt_init.__name__: SqrtInit,
            sqrt.__name__: Sqrt,
            r_sqrt_init.__name__: RSqrtInit,
            r_sqrt.__name__: RSqrt,
            sigmoid_init.__name__: SigmoidInit,
            sigmoid.__name__: Sigmoid,
            log_init.__name__: LogInit,
            log.__name__: Log,
            log_with_base_init.__name__: LogWithBaseInit,
            log_with_base.__name__: LogWithBase,
            power_init.__name__: PowerInit,
            power.__name__: Power,
            r_sub_init.__name__: RSubInit,
            r_sub.__name__: RSub,
            sign_bit_init.__name__: SignBitInit,
            sign_bit.__name__: SignBit,
            square_init.__name__: SquareInit,
            square.__name__: Square,
            reduce.__name__: Reduce,
            transpose_wh_init.__name__: TransposeWHInit,
            transpose_wh.__name__: TransposeWH,
            tanh_init.__name__: TanhInit,
            tanh.__name__: Tanh,
            tan_init.__name__: TanInit,
            tan.__name__: Tan,
            sin_init.__name__: SinInit,
            sin.__name__: Sin,
            cos_init.__name__: CosInit,
            cos.__name__: Cos,
            asin_init.__name__: AsinInit,
            asin.__name__: Asin,
            atan_init.__name__: AtanInit,
            atan.__name__: Atan,
            acos_init.__name__: AcosInit,
            acos.__name__: Acos,
            ltz_init.__name__: LtzInit,
            ltz.__name__: Ltz,
            eqz_init.__name__: EqzInit,
            eqz.__name__: Eqz,
            lez_init.__name__: LezInit,
            lez.__name__: Lez,
            gtz_init.__name__: GtzInit,
            gtz.__name__: Gtz,
            gez_init.__name__: GezInit,
            gez.__name__: Gez,
            nez_init.__name__: NezInit,
            nez.__name__: Nez,
            unary_ne_init.__name__: UnaryNeInit,
            unary_ne.__name__: UnaryNe,
            unary_gt_init.__name__: UnaryGtInit,
            unary_gt.__name__: UnaryGt,
            unary_lt_init.__name__: UnaryLtInit,
            unary_lt.__name__: UnaryLt,
            tilize_init.__name__: TilizeInit,
            tilize_init_short.__name__: TilizeInitShort,
            tilize_init_short_with_dt.__name__: TilizeInitShortWithDT,
            tilize_block.__name__: TilizeBlock,
            tilize_uninit.__name__: TilizeUninit,
            tilize_uninit_with_dt.__name__: TilizeUninitWithDT,
            untilize_init.__name__: UntilizeInit,
            untilize_init_short.__name__: UntilizeInitShort,
            untilize_block.__name__: UntilizeBlock,
            untilize_uninit.__name__: UntilizeUninit,
        }

    def get_type(self, variable_name: str):
        return self.type_checker.types[variable_name]

    def get_operation(self, node, mlir_type) -> type:
        assert isinstance(node, ast.BinOp)
        return self._operations[mlir_type][type(node.op)]

    def generic_visit(self, node):
        print("Missing handling for: " + node.__class__.__name__)
        raise Exception(
            f"Unhandled construct, no parser provided: {node.__class__.__name__}"
        )

    def visit_Import(self, node) -> List[Operation]:
        return []

    def visit_ImportFrom(self, node):
        return []

    def visit_Pass(self, node) -> List[Operation]:
        return []

    def visit_Module(self, node) -> List[Operation]:
        operations: List[Operation] = []

        # at the 'module' level should just be a single function def
        for child in node.body:
            ops = self.visit(child)
            if isa(ops, list):
                operations += ops
            else:
                operations.append(ops)

        # after all processing is done, wrap in module operation
        self.operations = builtin.ModuleOp(operations)
        return [self.operations]

    def visit_FunctionDef(self, node) -> Operation:
        operations: List[Operation] = []

        return_types = []

        arg_types = []
        arg_names = []
        for arg in node.args.args:
            assert arg.annotation.id in TYPE_STR_TO_MLIR_TYPE
            arg_types.append(TYPE_STR_TO_MLIR_TYPE[arg.annotation.id])
            arg_names.append(arg.arg)

        block = Block(arg_types=arg_types)
        # Need to improve here with scoping, it's OK for now but will need to nest this
        # in order to push and pop when handling functions
        for index, arg_name in enumerate(arg_names):
            self.symbol_table[arg_name] = block.args[index]

        decorator_name = None
        if len(node.decorator_list) == 1:
            decorator_name = node.decorator_list[0].attr

        # set the current scope
        self.operations = operations

        # TODO: reconsider kernel name, fixed or user-defined?
        fn_name = node.name
        if decorator_name == "data_in":
            fn_name = "kernel_main"
            self.fn_kernel_type = KernelType.DATA_IN
        elif decorator_name == "host":
            fn_name = "main"
            self.fn_kernel_type = KernelType.HOST

        operations = self.generate_body_ops(node)

        if self.fn_kernel_type == KernelType.HOST:
            return_types.append(i32)
            zero_c = arith.ConstantOp(IntegerAttr(0, i32))
            ret = func.ReturnOp(zero_c)
            operations += [zero_c, ret]
        else:
            operations.append(func.ReturnOp())

        block.add_ops(list(flatten(operations)))
        region = Region(block)

        func_op = func.FuncOp(
            fn_name, FunctionType.from_lists(arg_types, return_types), region
        )

        self.fn_kernel_type = None

        # return some function definition with contents
        return builtin.ModuleOp(
            [func_op], {"kernel_type": builtin.StringAttr(decorator_name)}
        )

    def visit_Assign(self, node) -> Tuple[List[Operation], SSAValue]:
        # visit RHS, e.g. Constant in 'a = 0'
        rhs_ops, rhs_ssa_val = self.visit(node.value)
        operations = rhs_ops

        # get the variable name to store the result in
        dest = node.targets[0]
        if isa(dest, ast.Name):
            # Scalar assignment
            var_name = dest.id
            target_type = self.type_checker.types[var_name]
        elif isa(dest, ast.Subscript):
            # Array assignment
            var_name = dest.value.id
            target_type = self.type_checker.types[var_name].element_type
        else:
            assert False

        # if the types don't match we need to insert a cast operation
        operations, rhs_ssa_val = cast_if_needed(rhs_ssa_val, target_type, operations)

        if isa(dest, ast.Name):
            # create a memref
            # seen = var_name in self.symbol_table
            seen = var_name in self.symbol_table
            last_op = rhs_ops[-1]

            # TODO: not software engineering -- fix at a later date
            # TODO: check the fact of only
            # we [only] use builtin.UnrealizedConversionCastOp to model some MemRef
            # allocations, so look out for that (I think)
            if (
                isa(last_op, memref.AllocaOp)
                or isa(last_op, memref.AllocOp)
                or isa(last_op, builtin.UnrealizedConversionCastOp)
            ):
                # This is a bit of a hack, we are allocating a list and this is done
                # in the bin op, so we pick the memref here
                self.symbol_table[var_name] = rhs_ops[-1].results[0]
                rhs_ops[-1].results[0].name_hint = var_name
                assert not seen
                location = self.symbol_table[var_name]
            else:
                # This is a normal value, so just store as usual
                if not seen:
                    operations += [self.allocate_memory(var_name)]

                location = self.symbol_table[var_name]

                # store result in that memref
                store = memref.StoreOp.get(rhs_ssa_val, location, [])
                operations.append(store)
            return operations, location
        elif isa(dest, ast.Subscript):
            # Array assignment
            idx_ops, idx_ssa = self.visit(dest.slice)
            assert var_name in self.symbol_table
            if isa(idx_ssa.type, builtin.IntegerType):
                index_cast = arith.IndexCastOp(idx_ssa, builtin.IndexType())
                idx_ops.append(index_cast)
                idx_ssa = index_cast.results[0]
            store = memref.StoreOp.get(
                rhs_ssa_val, self.symbol_table[var_name], [idx_ssa]
            )
            return operations + idx_ops + [store], self.symbol_table[var_name]

    def visit_Constant(self, node) -> Tuple[List[Operation], SSAValue]:
        data = node.value

        arith_op = None

        if isinstance(data, bool):
            arith_op = arith.ConstantOp(BoolAttr(data, IntegerType(1)))
        elif isinstance(data, int):
            arith_op = arith.ConstantOp(IntegerAttr(data, IntegerType(32)))
        elif isinstance(data, float):
            arith_op = arith.ConstantOp(FloatAttr(data, Float32Type()))

        if arith_op is None:
            raise Exception(f"Unhandled constant type: {data.__class__.__name__}")

        return [arith_op], arith_op.results[0]

    def visit_List(self, node: ast.List):
        assert len(node.elts) == 1
        element_type_ops, element_type_ssa = self.visit(node.elts[0])
        return None, MemRefType(element_type_ssa.type, [])

    def visit_BinOp(self, node: ast.BinOp) -> List[Operation]:
        lhs_ops, lhs_ssa_val = self.visit(node.left)
        rhs_ops, rhs_ssa_val = self.visit(node.right)

        if isa(lhs_ssa_val, MemRefType):
            # This is a list definition, grab the size on the RHS and use memref alloc
            if isa(rhs_ops[0], arith.ConstantOp):
                assert isa(rhs_ops[0].value.type, IntegerType)
                size_val = rhs_ops[0].value.value.data
                alloc_shape = [size_val]
                dynamic_size = []
            else:
                alloc_shape = [-1]
                dynamic_size = [rhs_ssa_val]

            assert self.fn_kernel_type is not None
            if self.fn_kernel_type == KernelType.HOST:
                # On the host use an alloc to allocate on the heap, use the stack on the device in a kernel
                memory = memref.AllocOp.get(
                    lhs_ssa_val.element_type,
                    shape=alloc_shape,
                    dynamic_sizes=dynamic_size,
                )
            else:
                memory = memref.AllocaOp.get(
                    lhs_ssa_val.element_type,
                    shape=alloc_shape,
                    dynamic_sizes=dynamic_size,
                )
            operations = rhs_ops + [memory]
            return operations, memory.results[0]
        else:
            operations = lhs_ops + rhs_ops

            # if types differ, we need to cast for the operation
            target_type = self.type_checker.dominating_type(
                lhs_ssa_val.type,
                rhs_ssa_val.type,
            )

            operations, lhs_ssa_val = cast_if_needed(
                lhs_ssa_val, target_type, operations
            )
            operations, rhs_ssa_val = cast_if_needed(
                rhs_ssa_val, target_type, operations
            )

            # special case: if we have a division, we also want to cast
            if isinstance(node, ast.Div):
                target_type = Float32Type()
                operations, lhs_ssa_val = cast_if_needed(
                    lhs_ssa_val, target_type, operations
                )
                operations, rhs_ssa_val = cast_if_needed(
                    rhs_ssa_val, target_type, operations
                )

            op_constructor = self.get_operation(node, lhs_ssa_val.type)
            bin_op = op_constructor(lhs_ssa_val, rhs_ssa_val, None)
            return operations + [bin_op], bin_op.results[0]

    def visit_Subscript(self, node: ast.Subscript):
        from_location = self.symbol_table[node.value.id]
        idx_ops, idx_ssa = self.visit(node.slice)
        if isa(idx_ssa.type, builtin.IntegerType):
            index_cast = arith.IndexCastOp(idx_ssa, builtin.IndexType())
            idx_ops.append(index_cast)
            idx_ssa = index_cast.results[0]
        load = memref.LoadOp.get(from_location, [idx_ssa])
        return idx_ops + [load], load.results[0]

    def visit_Name(self, node: ast.Name) -> Tuple[List[Operation], SSAValue]:
        from_location = self.symbol_table[node.id]
        if isa(from_location.type, MemRefType):
            if len(from_location.type.shape) == 0:
                # If this is a scalar then load it from the memref
                load = memref.LoadOp.get(from_location, [])
                return [load], load.results[0]
            else:
                # If it is an array then return directly as we don't have array indexes
                # because if we did then this would be a subscript instead
                return [], from_location
        else:
            # If this is not a memref then just return it, it's likely a block argument
            # that is specifically typed e.g. an argument to the function
            return [], from_location

    def visit_If(self, node: ast.If) -> Tuple[List[Operation], Optional[SSAValue]]:
        allocations = self.allocate_new_variables(node)
        body_ops = self.generate_body_ops(node)

        condition_expr_ops, condition_ssa_val = self.visit(node.test)

        or_else = Region()

        if node.orelse:
            if isinstance(node.orelse[0], ast.If):
                or_else_ops, or_else_ssa_val = self.visit_If(node.orelse[0])
                or_else_ops = list(flatten(or_else_ops))
                or_else = Region(Block(or_else_ops))
            else:
                or_else = []
                for stmt in node.orelse:
                    or_else += self.visit(stmt)[0]

        # condition: SSAValue | Operation
        # return_types: Sequence[Attribute],
        # true_region: Region | Sequence[Block] | Sequence[Operation]
        # false_region: Region | Sequence[Block] | Sequence[Operation]
        if_statement = scf.IfOp(condition_ssa_val, [], body_ops, or_else)

        ssa_val = if_statement.results[0] if if_statement.results else None

        return allocations + condition_expr_ops + [if_statement], ssa_val

    def visit_Compare(self, node) -> Tuple[List[Operation], SSAValue]:
        left, l_val = self.visit(node.left)
        right, r_val = self.visit(node.comparators[0])

        operations = left + right

        comparison_op = node.ops[0]
        ideal_type = self.type_checker.dominating_type(
            l_val.type,
            r_val.type,
        )

        operations, l_val = cast_if_needed(l_val, ideal_type, operations)
        operations, r_val = cast_if_needed(r_val, ideal_type, operations)

        # TODO: handle sint, float comparisons
        op = self._uint_comparison[type(comparison_op)]
        operation = arith.CmpiOp(l_val, r_val, op)

        return operations + [operation], operation.results[0]

    def visit_For(self, node: ast.For) -> Tuple[List[Operation], Optional[SSAValue]]:
        from_expr, from_ssa = self.visit(node.iter.args[0])
        to_expr, to_ssa = self.visit(node.iter.args[1])

        # lb, up, step, iteration arguments, body
        step = arith.ConstantOp(IntegerAttr(1, IntegerType(32)))

        # adds variables to the symbol table and allocates memory for them
        var_allocations = self.allocate_new_variables(node)

        block = Block(arg_types=[i32])

        body = Region()
        body.add_block(block)

        for_loop = scf.ForOp(from_ssa, to_ssa, step.results[0], [], body)

        # if going for correct python semantics should probably manually
        # allocate space for the loop variable, and load into it for the first
        # instruction of the loop.
        iter_var_name = node.target.id
        alloc = self.allocate_memory(iter_var_name)

        loop_variable = for_loop.body.block.args[0]
        store = memref.StoreOp.get(loop_variable, alloc, [])

        loop_body_ops = [store] + self.generate_body_ops(node) + [scf.YieldOp()]
        block.add_ops(loop_body_ops)

        ssa_value = for_loop.results[0] if for_loop.results else None

        return var_allocations + [from_expr, to_expr, step, alloc, for_loop], ssa_value

    def visit_BoolOp(self, node: ast.BoolOp) -> Tuple[List[Operation], SSAValue]:
        # leftmost evaluation first
        # if a and b or c => if (a and b) or c
        lhs_ops, lhs_ssa_val = self.visit(node.values[0])  # a and b
        rhs_ops, rhs_ssa_val = self.visit(node.values[1])  # c

        match type(node.op):
            case ast.And:
                op = arith.AndIOp

            case ast.Or:
                op = arith.OrIOp

            case _:
                raise NotImplementedError(f"{node.op.__class__.__name__}")

        operation = op(lhs_ssa_val, rhs_ssa_val)

        return lhs_ops + rhs_ops + [operation], operation.results[0]

    def visit_UnaryOp(self, node) -> Tuple[List[Operation], SSAValue]:
        expr_ops, expr_ssa_val = self.visit(node.operand)
        true_decl = arith.ConstantOp(IntegerAttr.from_int_and_width(1, 1))

        match type(node.op):
            case ast.Not:
                unary_op = arith.XOrIOp(expr_ssa_val, true_decl.results[0])
                return expr_ops + [true_decl, unary_op], unary_op.results[0]
            case ast.USub:
                zero = arith.ConstantOp(IntegerAttr(0, IntegerType(32)))
                unary_op = arith.SubiOp(zero.results[0], expr_ssa_val)
                return expr_ops + [zero, unary_op], unary_op.results[0]
            case _:
                raise NotImplementedError(f"{node.op.__class__.__name__}")

    def visit_Expr(self, node) -> Tuple[List[Operation], SSAValue]:
        return self.visit(node.value)

    def handle_create_kernel(self, node) -> Tuple[List[Operation], SSAValue]:
        argc = len(node.args)
        if argc == 5:
            return self.handle_create_dm_kernel(node)
        if argc == 7:
            return self.handle_create_comp_kernel(node)

        raise TypeError(
            f"Expected 5 or 7 arguments for call to tt.Kernel(), found {argc}"
        )

    def handle_create_dm_kernel(self, node) -> Tuple[List[Operation], SSAValue]:
        program_ops, program_ssa = self.visit(node.args[0])
        core_ops, core_ssa = self.visit(node.args[2])

        assert isa(node.args[1], ast.Name)
        target_fn_name = node.args[1].id

        assert isa(node.args[3], ast.Attribute)

        rv_core_flag = None
        if node.args[3].attr == "DataMovement_0":
            rv_core_flag = RISCVCoreFlags.DATAMOVEMENT_0
        elif node.args[3].attr == "DataMovement_1":
            rv_core_flag = RISCVCoreFlags.DATAMOVEMENT_1
        else:
            assert False

        assert rv_core_flag is not None

        assert isa(node.args[4], ast.Constant)
        noc_id = node.args[4].value
        assert noc_id == 0 or noc_id == 1

        kernel_name = (
            "reader" if rv_core_flag == RISCVCoreFlags.DATAMOVEMENT_0 else "writer"
        )

        kernel_create = TTCreateKernel(
            program_ssa,
            core_ssa,
            f"{kernel_name}.cpp",
            RISCVCoreFlagsAttr([rv_core_flag]),
            noc_id,
        )

        return program_ops + core_ops + [kernel_create], kernel_create.results[0]

    def handle_create_comp_kernel(self, node) -> Tuple[List[Operation], SSAValue]:
        program_ops, program_ssa = self.visit(node.args[0])
        core_ops, core_ssa = self.visit(node.args[2])

        assert isa(node.args[1], ast.Name)

        mf_flag = MathFidelityFlags(node.args[3].attr)
        fp32_acc = node.args[4].value
        math_apx = node.args[5].value

        assert fp32_acc in [0, 1]
        assert math_apx in [0, 1]

        kernel_create = TTCreateComputeKernel(
            program_ssa,
            core_ssa,
            "compute.cpp",
            MathFidelityFlagsAttr([mf_flag]),
            IntegerAttr(fp32_acc, i1),
            IntegerAttr(math_apx, i1),
        )

        return program_ops + core_ops + [kernel_create], kernel_create.results[0]

    def handle_host_call(self, node, operation_class, expected_num_args):
        if expected_num_args is not None:
            assert len(node.args) == expected_num_args
        arg_ops = []
        arg_ssas = []
        for arg in node.args:
            ops, ssa = self.visit(arg)
            arg_ops += ops
            arg_ssas.append(ssa)

        operation = operation_class(*arg_ssas)
        if len(operation.results) > 0:
            return arg_ops + [operation], operation.results[0]
        else:
            return arg_ops + [operation], None

    def handle_create_cb_config(self, node):
        assert len(node.args) == 4

        num_pages_ops, num_pages_ssa = self.visit(node.args[0])
        page_size_ops, page_size_ssa = self.visit(node.args[1])
        cb_index_ops, cb_index_ssa = self.visit(node.args[2])
        assert isa(node.args[3], ast.Name)
        data_type = node.args[3].id

        cbConfig = TTCreateCBConfig(
            num_pages_ssa, page_size_ssa, cb_index_ssa, data_type
        )

        return num_pages_ops + page_size_ops + cb_index_ops + [
            cbConfig
        ], cbConfig.results[0]

    def handle_convert_to_array(self, node):
        source_ops, source_ssa = self.visit(node.args[0])
        assert isa(node.args[1], ast.Name)
        str_data_type = node.args[1].id
        assert isa(node.args[2], ast.Constant)
        data_size = node.args[2].value

        assert str_data_type in TYPE_STR_TO_MLIR_TYPE
        data_type = TYPE_STR_TO_MLIR_TYPE[str_data_type]

        target_memref = MemRefType(data_type, [data_size])
        conversion_op = builtin.UnrealizedConversionCastOp.get(
            [source_ssa], [target_memref]
        )
        return source_ops + [conversion_op], conversion_op.results[0]

    def handle_print(
        self, node: ast.Call
    ) -> Tuple[List[Operation], Optional[SSAValue]]:
        # ssa value of each arg
        # for now just handle string literals
        assert len(node.args) == 1
        string = node.args[0].value
        print_op = printf.PrintFormatOp(string)

        return [print_op], None

    def visit_Call(self, node) -> Tuple[List[Operation], Optional[SSAValue]]:
        if isa(node.func, ast.Attribute):
            name = node.func.attr
            if name == Core.__name__:
                return self.handle_host_call(node, TTHostCore, 2)
            if name == "DRAMConfig":
                return self.handle_host_call(node, TTCreateDRAMConfig, 2)
            if name == "CreateBuffer":
                return self.handle_host_call(node, TTCreateBuffer, 1)
            if name == "CreateDevice":
                return self.handle_host_call(node, TTCreateDevice, 1)
            if name == "GetCommandQueue":
                return self.handle_host_call(node, TTGetCommandQueue, 1)
            if name == "EnqueueWriteBuffer":
                return self.handle_host_call(node, TTEnqueueWriteBuffer, 4)
            if name == "EnqueueReadBuffer":
                return self.handle_host_call(node, TTEnqueueReadBuffer, 4)
            if name == "CreateProgram":
                return self.handle_host_call(node, TTCreateProgram, 0)
            if name == "Kernel":
                return self.handle_create_kernel(node)
            if name == "SetRuntimeArgs":
                return self.handle_host_call(node, TTSetRuntimeArgs, None)
            if name == "EnqueueProgram":
                return self.handle_host_call(node, TTEnqueueProgram, 3)
            if name == "Finish":
                return self.handle_host_call(node, TTFinish, 1)
            if name == "CloseDevice":
                return self.handle_host_call(node, TTCloseDevice, 1)
            if name == "GetMemoryAddress":
                return self.handle_host_call(node, TTGetMemoryAddress, 1)
            if name == "CBConfig":
                return self.handle_create_cb_config(node)
            if name == "CreateCircularBuffer":
                return self.handle_host_call(node, TTCreateCircularBuffer, 3)
            if name == "cb_get_write_ptr":
                return self.handle_host_call(node, CBGetWritePointer, 1)
            if name == "cb_get_read_ptr":
                return self.handle_host_call(node, CBGetReadPointer, 1)
            if name == "to_array":
                return self.handle_convert_to_array(node)
        elif isa(node.func, ast.Name) and node.func.id == "print":
            return self.handle_print(node)
        else:
            name = node.func.id
            if name not in self._functions:
                raise NotImplementedError(f"Unhandled function {name}")

        properties = []
        single_property_funcs = [
            get_noc_addr_from_bank_id.__name__,
            exp.__name__,
            exp_init.__name__,
            erf_init.__name__,
            erfc_init.__name__,
            erf.__name__,
            erfc.__name__,
            gelu.__name__,
            gelu_init.__name__,
            r_sqrt_init.__name__,
            r_sqrt.__name__,
            untilize_block.__name__,
            pack_tile.__name__,
            InterleavedAddrGen.__name__,
        ]

        # TODO: ideally want something more like:
        #   for i, prop_def in enumerate(Class.get_irdl_def().properties)
        #       properties.append(node.args.pop(i).value, prop_def.type)
        #  also allows us to remove single_property_funcs being hardcoded
        #  could construct dummy op and then throw away... thats solution for casting
        if name in single_property_funcs:
            first_arg = node.args.pop(0)
            if name == untilize_block.__name__:
                properties.append(IntegerAttr(first_arg.value, i32))

            # handle constexpr terms which are known at compile-time but not at
            # our level of compile-time
            elif isinstance(first_arg, ast.Name):
                pass

            else:
                properties.append(IntegerAttr(first_arg.value, IntegerType(1)))

        # We evaluate args in Python order (programmer intention) and then swap
        # only the SSA results that are given to the operation to preserve semantics

        results = []
        operations = []
        # Need to generically look at argument and ensure that types match arg type and operand type in operation
        # Then a conversion would be inserted in the below if needed
        for idx, arg in enumerate(node.args):
            ops, ssa_val = self.visit(arg)
            operations += ops
            results.append(ssa_val)

        match name:
            case noc_async_write_multicast.__name__:
                results[4], results[5], results[6] = results[5], results[6], results[4]
            case noc_semaphore_set_multicast.__name__:
                results[3], results[4], results[5] = results[4], results[5], results[3]

        args = properties + results
        constructor = self._functions[name]
        operation = constructor(*args)

        type_cast_ops = self.fix_types(constructor, operation, properties, results)
        operation = type_cast_ops[-1]

        result = operation.results[0] if operation.results else None

        return operations + type_cast_ops, result

    def fix_types(self, constructor, operation, props, results) -> List[Operation]:
        # TODO: looks like this is a class method, check?
        op_def = operation.get_irdl_definition()
        type_cast_ops = []

        for i, operand in enumerate(op_def.operands):
            # handle optional operands -- although assumes they are always at end
            if i >= len(results):
                break

            ssa_val = results[i]
            runtime_type = ssa_val.type
            types = operand[1].constr.constr

            # get first match if many possible (same types work)
            if isinstance(types, AnyOf):
                matched = False
                for target_type in types.attr_constrs:
                    try:
                        ops, ssa = get_cast(ssa_val, target_type.attr)
                        matched = True
                        type_cast_ops += ops
                        results[i] = ssa
                        break
                    except NotImplementedError as e:
                        continue

                if not matched:
                    raise NotImplementedError(
                        f"Unable to cast from {runtime_type} to one of: {types}"
                    )

            if isinstance(types, EqAttrConstraint):
                type_cast_ops, results[i] = cast_if_needed(
                    ssa_val, types.attr, type_cast_ops
                )

        new_operation = constructor(*(props + results))
        return type_cast_ops + [new_operation]

    def generate_body_ops(self, node: NodeWithBody) -> List[Operation]:
        operations = []

        for statement in node.body:
            ops, res = self.visit(statement)
            operations.append(ops)

        return list(flatten(operations))

    def get_assigned_variables(self, statement: ast.stmt) -> List[str]:
        if isinstance(statement, ast.Assign):
            if isa(statement.targets[0], ast.Name):
                return [statement.targets[0].id]
            elif isa(statement.targets[0], ast.Subscript):
                return [statement.targets[0].value.id]
            else:
                assert False

        if isinstance(statement, ast.For | ast.If | ast.While):
            names = []

            for child_stmt in statement.body:
                names += self.get_assigned_variables(child_stmt)

            return names

        if isinstance(statement, ast.Expr):
            return []

        # could also handle: ast.With, ast.FuncDef
        construct = statement.__class__.__name__
        raise Exception(f"Unhandled construct to explore: ast.{construct}")

    def allocate_new_variables(self, node: NodeWithBody) -> List[Operation]:
        """
        In Python, variables declared/initialised in the loop body persist the
        loop itself - there is no nested scope. This method searches a loop or
        if statement body in order to find all these variables and allocate them
        before the scf.For (etc.) scope. Sets preserve order in CPython 3.7+.
        """
        found_variables = []
        for statement in node.body:
            found_variables += self.get_assigned_variables(statement)

        # remove any existing variables from fresh variables
        found_variables = remove_duplicates(found_variables)
        fresh_variables = subtract(found_variables, items=self.symbol_table)

        allocations = []
        for var in list(fresh_variables):
            memory = self.allocate_memory(var)
            allocations.append(memory)

        return allocations

    def allocate_memory(self, symbol: str) -> memref.AllocOp:
        memory = memref.AllocOp([], [], MemRefType(self.get_type(symbol), []))
        memory.results[0].name_hint = symbol
        self.symbol_table[symbol] = memory.results[0]
        return memory

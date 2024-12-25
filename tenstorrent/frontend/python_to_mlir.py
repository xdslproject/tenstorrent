from typing import Dict, List, Tuple, Optional

from xdsl.dialects import builtin, func, arith, memref, scf
from xdsl.dialects.builtin import (FunctionType,
                                   ModuleOp, IndexType, FloatAttr, Float32Type, BoolAttr)
from xdsl.ir import Region, Block
from xdsl.utils.hints import isa

from tenstorrent.dialects import *
from tenstorrent.utils import flatten, remove_duplicates, subtract
from .dummy import *
from .memref_context import MemrefContext
from .type_checker import MLIRType

NodeWithBody = ast.If | ast.For | ast.While | ast.FunctionDef


class PythonToMLIR(ast.NodeVisitor):
    """
    Parses a Python AST to create operations from the tt_data, tt_compute,
    scf, and memref dialects.
    """

    def __init__(self, type_checker):
        super().__init__()
        self.symbol_table = MemrefContext()  # variable names -> memref

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

        self._operations: Dict[MLIRType, Dict[type(ast.operator), type(IRDLOperation)]] = {
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
            }
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

            # TODO: Should separate into different classes here for compute
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

    def get_operation(self, node) -> type:
        assert isinstance(node, ast.BinOp)
        t = self.type_checker.visit(node)
        return self._operations[t][type(node.op)]

    def generic_visit(self, node):
        print("Missing handling for: " + node.__class__.__name__)
        raise Exception(f"Unhandled construct, no parser provided: {node.__class__.__name__}")

    def visit_Import(self, node) -> List[Operation]:
        return []

    def visit_Pass(self, node) -> List[Operation]:
        return []

    def visit_Module(self, node):
        # at the 'module' level should just be a single function def
        self.operations = self.visit(node.body[0])

    def visit_FunctionDef(self, node) -> Operation:
        operations: List[Operation] = []

        decorator_name = None
        if len(node.decorator_list) == 1:
            decorator_name = node.decorator_list[0].attr

        # set the current scope
        self.operations = operations

        operations = self.generate_body_ops(node)
        operations.append(func.ReturnOp())

        block = Block(list(flatten(operations)))
        region = Region(block)

        fn_name = node.name
        if decorator_name == "data_in":
            fn_name = "kernel_main"
        elif decorator_name == "host":
            fn_name = "main"

        func_op = func.FuncOp(
            fn_name,
            FunctionType.from_lists([], []),
            region
        )

        # return some function definition with contents
        return builtin.ModuleOp(
            [func_op],
            {"kernel_type": builtin.StringAttr(decorator_name)}
        )

    def visit_Assign(self, node) -> Tuple[List[Operation], OpResult]:
        # visit RHS, e.g. Constant in 'a = 0'
        rhs_ops, rhs_ssa_val = self.visit(node.value)
        operations = rhs_ops

        # get the variable name to store the result in
        dest = node.targets[0]
        assert isinstance(dest, ast.Name)
        var_name = dest.id

        # create a memref
        # seen = var_name in self.symbol_table
        seen = var_name in self.symbol_table.dictionary
        location = self.allocate_memory(var_name) if not seen else self.symbol_table[var_name]

        # if the types don't match we need to insert a cast operation
        target_type = self.type_checker.types[var_name]
        if target_type != rhs_ssa_val.type:
            cast_ops, cast_ssa = self.get_cast(target_type, rhs_ssa_val)
            if len(cast_ops) > 0:
                assert cast_ssa is not None
                rhs_ssa_val = cast_ssa
                operations += cast_ops
            else:
                assert cast_ssa is None

        if not seen:
            operations += [location]

        # store result in that memref
        store = memref.StoreOp.get(rhs_ssa_val, location, [])

        return operations + [store], location.results[0]

    def get_cast(self, target_type: MLIRType, ssa_val: SSAValue) -> Tuple[List[Operation], Optional[OpResult]]:
        if isinstance(ssa_val.type, IntegerType):
            conv_op = None
            match target_type:
                case Float32Type():
                    conv_op = arith.ExtFOp(ssa_val, Float32Type())

                case IndexType():
                    conv_op = arith.IndexCastOp(ssa_val, IndexType())
            if conv_op is not None:
                return [conv_op], conv_op.results[0]
        return [], None

    def visit_Constant(self, node) -> Tuple[List[Operation], OpResult]:
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

    def visit_BinOp(self, node: ast.BinOp) -> Tuple[List[Operation], OpResult]:
        lhs, lhs_ssa_val = self.visit(node.left)
        rhs, rhs_ssa_val = self.visit(node.right)

        # need to insert operations at evaluation order:
        for op in lhs + rhs:
            if op not in self.operations:
                self.operations.append(op)

        operations = lhs + rhs

        # if types differ, we need to cast for the operation
        if lhs_ssa_val.type != rhs_ssa_val.type:
            target_type = self.type_checker.dominating_type(lhs_ssa_val.type, rhs_ssa_val.type)
            if lhs_ssa_val.type != target_type:
                l_cast_ops, lhs_ssa_val = self.get_cast(target_type, lhs_ssa_val)
                operations += l_cast_ops

            if rhs_ssa_val.type != target_type:
                r_cast_ops, rhs_ssa_val = self.get_cast(target_type, rhs_ssa_val)
                operations += r_cast_ops

        # special case: if we have a division, we also want to cast
        if isinstance(node, ast.Div):
            target_type = Float32Type()
            if lhs_ssa_val.type != target_type:
                l_cast_ops, lhs_ssa_val = self.get_cast(target_type, lhs_ssa_val)
                operations += l_cast_ops

            if rhs_ssa_val.type != target_type:
                r_cast_ops, rhs_ssa_val = self.get_cast(target_type, rhs_ssa_val)
                operations += r_cast_ops

        op_constructor = self.get_operation(node)
        bin_op = op_constructor(
            lhs_ssa_val,
            rhs_ssa_val,
            None
        )
        return operations + [bin_op], bin_op.results[0]

    def visit_Name(self, node: ast.Name) -> Tuple[List[Operation], OpResult]:
        from_location = self.symbol_table[node.id]
        load = memref.LoadOp.get(from_location, [])
        return [load], load.results[0]

    def visit_If(self, node: ast.If) -> Tuple[List[Operation], Optional[OpResult]]:
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
        if_statement = scf.IfOp(
            condition_ssa_val,
            [],
            body_ops,
            or_else
        )

        ssa_val = if_statement.results[0] if if_statement.results else None

        return allocations + condition_expr_ops + [if_statement], ssa_val

    def visit_Compare(self, node) -> Tuple[List[Operation], OpResult]:
        left, l_val = self.visit(node.left)
        right, r_val = self.visit(node.comparators[0])

        # TODO: handle sint, float comparisons
        op = self._uint_comparison[type(node.ops[0])]
        operation = arith.CmpiOp(l_val, r_val, op)

        return [left, right, operation], operation.results[0]

    def visit_For(self, node: ast.For) -> Tuple[List[Operation], Optional[OpResult]]:
        from_expr, from_ssa = self.visit(node.iter.args[0])
        to_expr, to_ssa = self.visit(node.iter.args[1])

        # lb, up, step, iteration arguments, body
        step = arith.ConstantOp(IntegerAttr(1, IntegerType(32)))

        # adds variables to the symbol table and allocates memory for them
        var_allocations = self.allocate_new_variables(node)

        block = Block(arg_types=[IndexType()])

        body = Region()
        body.add_block(block)

        for_loop = scf.ForOp(
            from_ssa,
            to_ssa,
            step.results[0],
            [],
            body
        )

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

    def visit_BoolOp(self, node: ast.BoolOp) -> Tuple[List[Operation], OpResult]:
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

        operation = op(
            lhs_ssa_val,
            rhs_ssa_val
        )

        return lhs_ops + rhs_ops + [operation], operation.results[0]

    def visit_UnaryOp(self, node) -> Tuple[List[Operation], OpResult]:
        expr_ops, expr_ssa_val = self.visit(node.operand)
        true_decl = arith.ConstantOp(IntegerAttr.from_int_and_width(1, 1))

        match type(node.op):
            case ast.Not:
                unary_op = arith.XOrIOp(expr_ssa_val, true_decl.results[0])
                return expr_ops + [
                    true_decl,
                    unary_op], unary_op.results[0]
            case ast.USub:
                zero = arith.ConstantOp(IntegerAttr(0, IntegerType(32)))
                unary_op = arith.SubiOp(zero.results[0], expr_ssa_val)
                return expr_ops + [
                    zero,
                    unary_op], unary_op.results[0]
            case _:
                raise NotImplementedError(f"{node.op.__class__.__name__}")

    def visit_Expr(self, node) -> Tuple[List[Operation], OpResult]:
        return self.visit(node.value)

    def handleCore(self, node):
        assert len(node.args) == 2
        x_core_ops, x_core_ssa_val = self.visit(node.args[0])
        y_core_ops, y_core_ssa_val = self.visit(node.args[1])

        core_create = TTHostCore(x_core_ssa_val, y_core_ssa_val)
        return x_core_ops + y_core_ops + [core_create], core_create.results[0]

    def visit_Call(self, node) -> Tuple[List[Operation], Optional[OpResult]]:
        if isa(node.func, ast.Attribute):
            name = node.func.attr
            if name == "Core":
                return self.handleCore(node)
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
        ]

        # TODO: object to count props and their types?
        #     must be a way to inspect the class vars...

        if name in single_property_funcs:
            if name == untilize_block.__name__:
                properties.append(IntegerAttr(node.args.pop(0).value, i32))
            else:
                properties.append(IntegerAttr(node.args.pop(0).value, IntegerType(1)))

        # We evaluate args in Python order (programmer intention) and then swap
        # only the SSA results that are given to the operation to preserve semantics
        arg_evals = list(flatten([self.visit(arg) for arg in node.args]))
        operations = list(flatten(map(lambda x: x[0], arg_evals)))
        results = list(map(lambda x: x[1], arg_evals))

        match name:
            case noc_async_write_multicast.__name__:
                results[4], results[5], results[6] = results[5], results[6], results[4]
            case noc_semaphore_set_multicast.__name__:
                results[3], results[4], results[5] = results[4], results[5], results[3]

        args = properties + results
        operation = self._functions[name](*args)
        result = operation.results[0] if operation.results else None

        return operations + [operation], result

    def generate_body_ops(self, node: NodeWithBody) -> List[Operation]:
        operations = []

        for statement in node.body:
            ops, res = self.visit(statement)
            operations.append(ops)

        return list(flatten(operations))


    def get_assigned_variables(self, statement: ast.stmt) -> List[str]:
        if isinstance(statement, ast.Assign):
            return [statement.targets[0].id]

        if isinstance(statement, ast.For | ast.If | ast.While):
            names = []

            for child_stmt in statement.body:
                names += self.get_assigned_variables(child_stmt)

            return names

        # could also handle: ast.With, ast.FuncDef
        construct = statement.__class__.__name__
        raise Exception(f"Unhandled construct to explore: {construct}")

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
        fresh_variables = subtract(found_variables, items=self.symbol_table.dictionary)

        allocations = []
        for var in list(fresh_variables):
            memory = self.allocate_memory(var)
            allocations.append(memory)

        return allocations

    def allocate_memory(self, symbol: str) -> memref.AllocOp:
        memory = memref.AllocOp([], [], MemRefType(self.get_type(symbol), []))
        memory.results[0].name_hint = symbol
        self.symbol_table[symbol] = memory
        return memory

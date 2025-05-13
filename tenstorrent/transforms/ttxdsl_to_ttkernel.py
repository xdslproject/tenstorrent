from typing import Type, Dict, Tuple, List

from xdsl.context import Context
from xdsl.dialects import builtin, memref
from xdsl.dialects.builtin import Float32Type, FunctionType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import BlockArgument, Block
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriter,
)

from tenstorrent.dialects import *


class ReplaceTTxOps(RewritePattern):
    def __init__(self):
        super().__init__()
        self.cb_port_ctr = 0
        self._block_arg_index = 0
        self.cb_port_type = "in"
        self.handled_ops: Dict[Type[Operation], Type[IRDLOperation]] = {
            compute.BinaryOpInitCommon: ttkernel.BinaryOpInitCommonOp,
            compute.AddInit: ttkernel.AddTilesInitOp,
            circular_buffer.CBWaitFront: ttkernel.CBWaitFrontOp,
            compute.RegsAcquire: ttkernel.TileRegsAcquireOp,
            compute.Add: ttkernel.AddTilesOp,
            compute.RegsCommit: ttkernel.TileRegsCommitOp,
            compute.RegsWait: ttkernel.TileRegsWaitOp,
            compute.PackTile: ttkernel.PackTileOp,
            compute.RegsRelease: ttkernel.TileRegsReleaseOp,
            circular_buffer.CBPopFront: ttkernel.CBPopFrontOp,
            circular_buffer.CBPushBack: ttkernel.CBPushBackOp,
        }

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter):
        for operation in op.walk():
            # TODO: this would currently break on nested funcs, fix for that
            if isinstance(operation, FuncOp):
                self.replace_func_op(operation, rewriter)

    def replace_func_op(self, func: FuncOp, rewriter: PatternRewriter):
        # prepare to replace this FuncOp
        replaced_ops: Dict[Operation, int] = {}

        for operation in func.walk():
            if isinstance(operation, ReturnOp):
                region = func.regions[0]
                func.detach_region(region)
                new_func_op = FuncOp(
                    func.sym_name,
                    FunctionType.from_lists(region.blocks.first.arg_types, []),
                    region,
                    func.sym_visibility,
                )
                rewriter.replace_op(func, new_func_op)
                return

            if isinstance(operation, FuncOp) and not operation == func:
                self.replace_func_op(operation, rewriter)

            if type(operation) in self.handled_ops:
                cur_op_type = type(operation)
                new_op_type = self.handled_ops[cur_op_type]
                op_def = new_op_type.get_irdl_definition()
                num_args = len(op_def.operands)

                new_op = new_op_type(*operation.operands[:num_args])

                rewriter.replace_op(operation, new_op, operation.results)
                for (name, param), arg in zip(op_def.operands, operation.operands):
                    if param.constr.constr.attr == CBType:
                        # TODO: use a CBType not an i32
                        # memref<8x4x4x1024xf32>
                        default_cb_type = CBType(
                            self.get_next_cb_port(),
                            IntegerAttr(0, 32),
                            MemRefType(Float32Type(), [8, 4, 4, 1024]),
                        )
                        block = func.body.block

                        # always expect our arguments' owners to be Operations not block arguments
                        # TODO: this isn't true right? think about Nick's data_in example with args
                        if arg.owner not in replaced_ops and isinstance(
                            arg.owner, Operation
                        ):
                            insert_arg_index = len(block.args)
                            replaced_ops[arg.owner] = insert_arg_index
                            block.insert_arg(default_cb_type, insert_arg_index)
                            rewriter.replace_op(
                                arg.owner, [], [block.args[insert_arg_index]]
                            )

                        # TODO: here arg.owner is a block argument, just change
                        #  the type from (e.g.) i32 -> CBType if needed
                        elif arg.owner not in replaced_ops and arg not in block.args:
                            raise NotImplementedError("found block arg as non-CBType")

    def get_next_cb_port(self):
        identifier = f"cb_{self.cb_port_type}{self.cb_port_ctr}"
        flag = next((f for f in CBPortFlags if f.value == identifier))
        cb_port = CBPortAttr([CBPortFlagsAttrBase([flag])])

        # set up the next cb port
        self.cb_port_ctr += 1
        if self.cb_port_ctr == 8:
            self.cb_port_ctr = 0
            self.set_next_cb_port_type()

        return cb_port

    def set_next_cb_port_type(self):
        types = ["in", "dataflow", "out", "intermediate"]
        i = types.index(self.cb_port_type) + 1
        self.cb_port_type = types[i % len(types)]

    def get_block_arg_index(self):
        self._block_arg_index += 1
        return self._block_arg_index - 1


class RemoveAllAllocs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


class RemoveAllStores(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op([])


class RemoveAllLoads(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter):
        """
        Everytime we find a load, find its corresponding store and directly
        replace the result with that value. May not work for variables that
        change depending on how .uses works... (past uses vs all uses)
        """
        memref_ssa = op.operands[0]
        stored_val_ssa = None

        for use in memref_ssa.uses:
            if isinstance(use.operation, memref.StoreOp):
                stored_val_ssa = use.operation.operands[0]

        assert stored_val_ssa is not None
        rewriter.replace_matched_op([], [stored_val_ssa])


class RemoveAllUnrealizedConversionCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: builtin.UnrealizedConversionCastOp, rewriter: PatternRewriter
    ):
        rewriter.replace_matched_op([], [op.operands[0]])


class ConvertTTxToTTKernel(ModulePass):
    """
    Transforms code written in the Tenstorrent xDSL dialect to the officially
    supported ttmlir dialect written by Tenstorrent
    """

    name = "convert-ttx-to-ttkernel"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveAllUnrealizedConversionCasts(),
                    RemoveAllLoads(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveAllStores(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveAllAllocs(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ReplaceTTxOps(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

        # 'op' is the higest level op
        reg = op.regions[0]
        op.detach_region(reg)

        next_module = reg.first_block.first_op
        if isinstance(next_module, builtin.ModuleOp):
            next_reg = next_module.regions[0]
            next_module.detach_region(next_reg)
            op.add_region(next_reg)

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class ExtractMetalium(ModulePass):
    """
    Extract the Metalium-related code from code transformed to discover
    possible Metalium operations
    """

    name = "extract-metalium"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        reg = op.regions[0]

        # we assume there is 5 ops within the region (driver, host, *kernels)
        # and we want the final 4 as these can be compiled together. Later may
        # want to extract data_in, data_out, compute, host, separately
        assert len(reg.ops) == 5
        reg.block.detach_op(reg.block.first_op)

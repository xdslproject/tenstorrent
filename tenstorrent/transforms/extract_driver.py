from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin

from xdsl.passes import ModulePass


@dataclass(frozen=True)
class ExtractDriver(ModulePass):
    """
    After transformations which map the linear algebra dialect to Tenstorrent
    MLIR, we should have 5 builtin.ModuleOp operations:
    - Driver host
    - Metalium host
    - Metalium kernels (data in, compute, data out)

    This extracts the driver host, which is somewhat the "remaining code" that
    wasn't transformed into the Tenstorrent dialects
    """

    name = "extract-driver"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        reg = op.regions[0]
        op.detach_region(reg)

        first_module = reg.first_block.first_op
        assert isinstance(first_module, builtin.ModuleOp)
        assert "kernel_type" not in first_module.attributes
        assert "kernel_type" not in first_module.properties

        # replace the MLIR such that only that module exists
        target_region = first_module.regions[0]
        first_module.detach_region(target_region)
        op.add_region(target_region)

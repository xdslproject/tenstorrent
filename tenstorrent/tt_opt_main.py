import argparse
from typing import IO
from xdsl.dialects.builtin import ModuleOp
from typing import Callable, Dict, List
from xdsl.xdsl_opt_main import xDSLOptMain

from tenstorrent.dialects.ttshared import TTShared
from tenstorrent.dialects.data_movement import DataMovement
from tenstorrent.dialects.circular_buffer import CircularBuffer
from tenstorrent.dialects.compute import Compute
from tenstorrent.dialects.host import TTHost
from tenstorrent.backend.print_metalium import PrintMetalium
from tenstorrent.transforms.linalg_to_tt import RewriteMatmulToTT


class TTOptMain(xDSLOptMain):
    passes_native = []

    def register_all_passes(self):
        super().register_all_passes()
        self.register_pass(RewriteMatmulToTT.name, lambda: RewriteMatmulToTT)

    def register_all_targets(self):
        super().register_all_targets()

        def _output_metalium(prog: ModuleOp, output: IO[str]):
            printer = PrintMetalium()
            printer.print_op(prog)

        self.available_targets["tt-metalium"] = _output_metalium

    def setup_pipeline(self):
        super().setup_pipeline()

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)

    def register_all_dialects(self):
        super().register_all_dialects()
        self.ctx.load_dialect(DataMovement)
        self.ctx.load_dialect(CircularBuffer)
        self.ctx.load_dialect(TTHost)
        self.ctx.load_dialect(Compute)
        self.ctx.load_dialect(TTShared)

    @staticmethod
    def get_passes_as_dict() -> Dict[str, Callable[[ModuleOp], None]]:
        """Add all passes that can be called by tt-opt in a dictionary."""

        pass_dictionary = {}

        passes = TTOptMain.passes_native

        for pass_function in passes:
            pass_dictionary[pass_function.__name__.replace("_", "-")] = pass_function

        return pass_dictionary

    @staticmethod
    def get_passes_as_list(native=False, integrated=False) -> List[str]:
        """Add all passes that can be called by tt-opt in a dictionary."""

        pass_list = []

        passes = TTOptMain.passes_native

        for pass_function in passes:
            pass_list.append(pass_function.__name__.replace("_", "-"))

        return pass_list

    def register_all_frontends(self):
        super().register_all_frontends()

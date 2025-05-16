# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.data_in
def template_call():
    tt.exp(True, 3)


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main() {
# CHECK-NEXT:       %0 = arith.constant 3 : i32
# CHECK-NEXT:       %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
# CHECK-NEXT:       "comp.exp_tile"(%1) <{fast_and_approx = true}> : (ui32) -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

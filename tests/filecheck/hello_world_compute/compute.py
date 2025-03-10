# RUN: python3.13 %s | filecheck %s

from tests import run_from_examples

run_from_examples(__file__)


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "compute"} {
# CHECK-NEXT:     func.func @void_compute() {
# CHECK-NEXT:       printf.print_format "Hello, I am running a void compute kernel"
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

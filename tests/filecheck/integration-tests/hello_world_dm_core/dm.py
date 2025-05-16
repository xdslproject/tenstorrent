# RUN: python3.13 %s | filecheck %s

from tests import run_from_examples

run_from_examples(__file__)

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main() {
# CHECK-NEXT:       printf.print_format "Hello, I am running a void data movement kernel on NOC 0."
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_out"} {
# CHECK-NEXT:     func.func @void_dataflow1() {
# CHECK-NEXT:       printf.print_format "Hello, I am running a void data movement kernel on NOC 1."
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def print_host():
    print("Hello I am the host")


@tt.data_in
def print_d0():
    print("Hello, I am the data in core")


@tt.data_out
def print_d1():
    print("Hello, I am the data out core")


@tt.compute
def print_compute():
    print("Hello, I am the compute core")

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       printf.print_format "Hello I am the host"
# CHECK-NEXT:       %0 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %0 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main() {
# CHECK-NEXT:       printf.print_format "Hello, I am the data in core"
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_out"} {
# CHECK-NEXT:     func.func @print_d1() {
# CHECK-NEXT:       printf.print_format "Hello, I am the data out core"
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "compute"} {
# CHECK-NEXT:     func.func @print_compute() {
# CHECK-NEXT:       printf.print_format "Hello, I am the compute core"
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

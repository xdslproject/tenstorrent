# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.host
def host_api():
    device = tt.CreateDevice(0)


@tt.data_in
def data_in_api():
    tt.noc_async_read_barrier()


@tt.data_out
def data_out_api():
    tt.noc_async_write_barrier()


@tt.compute
def compute():
    tt.add(1, 2, 3, 4, 5)


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 0 : i32
# CHECK-NEXT:       %1 = "tthost.create_device"(%0) : (i32) -> !tthost.device
# CHECK-NEXT:       %device = memref.alloc() : memref<!tthost.device>
# CHECK-NEXT:       memref.store %1, %device[] : memref<!tthost.device>
# CHECK-NEXT:       %2 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %2 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main() {
# CHECK-NEXT:       "dm.noc_async_read_barrier"() : () -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_out"} {
# CHECK-NEXT:     func.func @data_out_api() {
# CHECK-NEXT:       "dm.noc_async_write_barrier"() : () -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   builtin.module attributes {kernel_type = "compute"} {
# CHECK-NEXT:     func.func @compute() {
# CHECK-NEXT:       %0 = arith.constant 1 : i32
# CHECK-NEXT:       %1 = arith.constant 2 : i32
# CHECK-NEXT:       %2 = arith.constant 3 : i32
# CHECK-NEXT:       %3 = arith.constant 4 : i32
# CHECK-NEXT:       %4 = arith.constant 5 : i32
# CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %0 : i32 to ui32
# CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %1 : i32 to ui32
# CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %2 : i32 to ui32
# CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %3 : i32 to ui32
# CHECK-NEXT:       %9 = builtin.unrealized_conversion_cast %4 : i32 to ui32
# CHECK-NEXT:       "comp.add_tiles"(%5, %6, %7, %8, %9) : (ui32, ui32, ui32, ui32, ui32) -> ()
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

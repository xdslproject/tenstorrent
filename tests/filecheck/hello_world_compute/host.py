# RUN: python3.13 %s | filecheck %s

from tests import run_from_examples

run_from_examples(__file__)

# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "host"} {
# CHECK-NEXT:     func.func @main() -> i32 {
# CHECK-NEXT:       %0 = arith.constant 0 : i32
# CHECK-NEXT:       %1 = arith.constant 0 : i32
# CHECK-NEXT:       %2 = "tthost.core"(%0, %1) : (i32, i32) -> !tthost.corecoord
# CHECK-NEXT:       %core = memref.alloc() : memref<!tthost.corecoord>
# CHECK-NEXT:       memref.store %2, %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %3 = arith.constant 0 : i32
# CHECK-NEXT:       %4 = "tthost.create_device"(%3) : (i32) -> !tthost.device
# CHECK-NEXT:       %device = memref.alloc() : memref<!tthost.device>
# CHECK-NEXT:       memref.store %4, %device[] : memref<!tthost.device>
# CHECK-NEXT:       %5 = memref.load %device[] : memref<!tthost.device>
# CHECK-NEXT:       %6 = "tthost.get_command_queue"(%5) : (!tthost.device) -> !tthost.command_queue
# CHECK-NEXT:       %cq = memref.alloc() : memref<!tthost.command_queue>
# CHECK-NEXT:       memref.store %6, %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %7 = "tthost.create_program"() : () -> !tthost.program
# CHECK-NEXT:       %program = memref.alloc() : memref<!tthost.program>
# CHECK-NEXT:       memref.store %7, %program[] : memref<!tthost.program>
# CHECK-NEXT:       %8 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %9 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %10 = "tthost.create_compute_kernel"(%8, %9) <{kernel_name = "compute.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<HiFi4>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
# CHECK-NEXT:       %void_compute_kernel = memref.alloc() : memref<!tthost.kernel>
# CHECK-NEXT:       memref.store %10, %void_compute_kernel[] : memref<!tthost.kernel>
# CHECK-NEXT:       %11 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %12 = memref.load %void_compute_kernel[] : memref<!tthost.kernel>
# CHECK-NEXT:       %13 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       "tthost.set_runtime_args"(%11, %12, %13) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
# CHECK-NEXT:       %14 = memref.load %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %15 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %16 = arith.constant false
# CHECK-NEXT:       "tthost.enqueue_program"(%14, %15, %16) : (!tthost.command_queue, !tthost.program, i1) -> ()
# CHECK-NEXT:       printf.print_format "Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication."
# CHECK-NEXT:       %17 = memref.load %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       "tthost.finish"(%17) : (!tthost.command_queue) -> ()
# CHECK-NEXT:       printf.print_format "Thank you, Core {0, 0} on Device 0, for the completed task"
# CHECK-NEXT:       %18 = memref.load %device[] : memref<!tthost.device>
# CHECK-NEXT:       "tthost.close_device"(%18) : (!tthost.device) -> ()
# CHECK-NEXT:       %19 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %19 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

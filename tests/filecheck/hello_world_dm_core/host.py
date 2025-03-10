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
# CHECK-NEXT:       %10 = "tthost.create_kernel"(%8, %9) <{kernel_name = "void_dataflow_kernel.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
# CHECK-NEXT:       %dataflow = memref.alloc() : memref<!tthost.kernel>
# CHECK-NEXT:       memref.store %10, %dataflow[] : memref<!tthost.kernel>
# CHECK-NEXT:       %11 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %12 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       %13 = "tthost.create_kernel"(%11, %12) <{kernel_name = "void_dataflow_kernel.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
# CHECK-NEXT:       %dataflow_1 = memref.alloc() : memref<!tthost.kernel>
# CHECK-NEXT:       memref.store %13, %dataflow_1[] : memref<!tthost.kernel>
# CHECK-NEXT:       %14 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %15 = memref.load %dataflow[] : memref<!tthost.kernel>
# CHECK-NEXT:       %16 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       "tthost.set_runtime_args"(%14, %15, %16) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
# CHECK-NEXT:       %17 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %18 = memref.load %dataflow_1[] : memref<!tthost.kernel>
# CHECK-NEXT:       %19 = memref.load %core[] : memref<!tthost.corecoord>
# CHECK-NEXT:       "tthost.set_runtime_args"(%17, %18, %19) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
# CHECK-NEXT:       %20 = memref.load %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       %21 = memref.load %program[] : memref<!tthost.program>
# CHECK-NEXT:       %22 = arith.constant false
# CHECK-NEXT:       "tthost.enqueue_program"(%20, %21, %22) : (!tthost.command_queue, !tthost.program, i1) -> ()
# CHECK-NEXT:       printf.print_format "Hello, Core {0, 0} on Device 0, I am sending you some data. Standby awaiting communication."
# CHECK-NEXT:       %23 = memref.load %cq[] : memref<!tthost.command_queue>
# CHECK-NEXT:       "tthost.finish"(%23) : (!tthost.command_queue) -> ()
# CHECK-NEXT:       printf.print_format "Thank you, Core {0, 0} on Device 0, for the completed task."
# CHECK-NEXT:       %24 = memref.load %device[] : memref<!tthost.device>
# CHECK-NEXT:       "tthost.close_device"(%24) : (!tthost.device) -> ()
# CHECK-NEXT:       %25 = arith.constant 0 : i32
# CHECK-NEXT:       func.return %25 : i32
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 0 : i32
      %2 = "tthost.core"(%0, %1) : (i32, i32) -> !tthost.corecoord
      %core = memref.alloc() : memref<!tthost.corecoord>
      memref.store %2, %core[] : memref<!tthost.corecoord>
      %3 = arith.constant 0 : i32
      %4 = "tthost.create_device"(%3) : (i32) -> !tthost.device
      %device = memref.alloc() : memref<!tthost.device>
      memref.store %4, %device[] : memref<!tthost.device>
      %5 = memref.load %device[] : memref<!tthost.device>
      %6 = "tthost.get_command_queue"(%5) : (!tthost.device) -> !tthost.command_queue
      %cq = memref.alloc() : memref<!tthost.command_queue>
      memref.store %6, %cq[] : memref<!tthost.command_queue>
      %7 = "tthost.create_program"() : () -> !tthost.program
      %program = memref.alloc() : memref<!tthost.program>
      memref.store %7, %program[] : memref<!tthost.program>
      %8 = memref.load %program[] : memref<!tthost.program>
      %9 = memref.load %core[] : memref<!tthost.corecoord>
      %10 = "tthost.create_kernel"(%8, %9) <{kernel_name = "void_dataflow_kernel.cpp", riscv_core = #tthost.riscv_core<datamovement_0>, noc_id = #builtin.int<0>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %dataflow = memref.alloc() : memref<!tthost.kernel>
      memref.store %10, %dataflow[] : memref<!tthost.kernel>
      %11 = memref.load %program[] : memref<!tthost.program>
      %12 = memref.load %core[] : memref<!tthost.corecoord>
      %13 = "tthost.create_kernel"(%11, %12) <{kernel_name = "void_dataflow_kernel.cpp", riscv_core = #tthost.riscv_core<datamovement_1>, noc_id = #builtin.int<1>}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %dataflow_1 = memref.alloc() : memref<!tthost.kernel>
      memref.store %13, %dataflow_1[] : memref<!tthost.kernel>
      %14 = memref.load %program[] : memref<!tthost.program>
      %15 = memref.load %dataflow[] : memref<!tthost.kernel>
      %16 = memref.load %core[] : memref<!tthost.corecoord>
      "tthost.set_runtime_args"(%14, %15, %16) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
      %17 = memref.load %program[] : memref<!tthost.program>
      %18 = memref.load %dataflow_1[] : memref<!tthost.kernel>
      %19 = memref.load %core[] : memref<!tthost.corecoord>
      "tthost.set_runtime_args"(%17, %18, %19) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
      %20 = memref.load %cq[] : memref<!tthost.command_queue>
      %21 = memref.load %program[] : memref<!tthost.program>
      %22 = arith.constant false
      "tthost.enqueue_program"(%20, %21, %22) : (!tthost.command_queue, !tthost.program, i1) -> ()
      printf.print_format "Hello, Core {0, 0} on Device 0, I am sending you some data. Standby awaiting communication."
      %23 = memref.load %cq[] : memref<!tthost.command_queue>
      "tthost.finish"(%23) : (!tthost.command_queue) -> ()
      printf.print_format "Thank you, Core {0, 0} on Device 0, for the completed task."
      %24 = memref.load %device[] : memref<!tthost.device>
      "tthost.close_device"(%24) : (!tthost.device) -> ()
      %25 = arith.constant 0 : i32
      func.return %25 : i32
    }
  }
}
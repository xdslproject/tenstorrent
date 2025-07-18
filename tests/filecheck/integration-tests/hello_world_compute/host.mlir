// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

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
      %10 = "tthost.create_compute_kernel"(%8, %9) <{kernel_name = "void_compute_kernel.cpp", riscv_core = #tthost.riscv_core<compute>, math_fidelity = #tthost.math_fidelity<HiFi4>, fp32_dest_acc_en = false, math_approx_mode = false}> : (!tthost.program, !tthost.corecoord) -> !tthost.kernel
      %void_compute_kernel = memref.alloc() : memref<!tthost.kernel>
      memref.store %10, %void_compute_kernel[] : memref<!tthost.kernel>
      %11 = memref.load %program[] : memref<!tthost.program>
      %12 = memref.load %void_compute_kernel[] : memref<!tthost.kernel>
      %13 = memref.load %core[] : memref<!tthost.corecoord>
      "tthost.set_runtime_args"(%11, %12, %13) {operandSegmentSizes = array<i32: 1, 1, 1, 0>} : (!tthost.program, !tthost.kernel, !tthost.corecoord) -> ()
      %14 = memref.load %cq[] : memref<!tthost.command_queue>
      %15 = memref.load %program[] : memref<!tthost.program>
      %16 = arith.constant false
      "tthost.enqueue_program"(%14, %15, %16) : (!tthost.command_queue, !tthost.program, i1) -> ()
      printf.print_format "Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication."
      %17 = memref.load %cq[] : memref<!tthost.command_queue>
      "tthost.finish"(%17) : (!tthost.command_queue) -> ()
      printf.print_format "Thank you, Core {0, 0} on Device 0, for the completed task"
      %18 = memref.load %device[] : memref<!tthost.device>
      "tthost.close_device"(%18) : (!tthost.device) -> ()
      %19 = arith.constant 0 : i32
      func.return %19 : i32
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     CoreCoord core = CoreCoord{0, 0};
// CHECK-NEXT:     IDevice* device_0 = CreateDevice(0);
// CHECK-NEXT:     IDevice* device = device_0;
// CHECK-NEXT:     CommandQueue & cq = device->command_queue();
// CHECK-NEXT:     Program program_1 = CreateProgram();
// CHECK-NEXT:     Program& program = program_1;
// CHECK-NEXT:     KernelHandle kernel_2 = CreateKernel(program, "void_compute_kernel.cpp", core, ComputeConfig {.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false, .compile_args = {}});
// CHECK-NEXT:     KernelHandle void_compute_kernel = kernel_2;
// CHECK-NEXT:     SetRuntimeArgs(program, void_compute_kernel, core, {});
// CHECK-NEXT:     EnqueueProgram(cq, program, false);
// CHECK-NEXT:     printf("Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");
// CHECK-NEXT:     Finish(cq);
// CHECK-NEXT:     printf("Thank you, Core {0, 0} on Device 0, for the completed task\n");
// CHECK-NEXT:     CloseDevice(device);
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }
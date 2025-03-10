// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "compute"} {
    func.func @void_compute() {
      printf.print_format "Hello, I am running a void compute kernel"
      func.return
    }
  }
}

// CHECK:      void void_compute() {
// CHECK-NEXT:     DPRINT_MATH(DPRINT << "Hello, I am running a void compute kernel" << ENDL());
// CHECK-NEXT: }
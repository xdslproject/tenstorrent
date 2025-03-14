// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s
builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      printf.print_format "Hello I am the host"
      %0 = arith.constant 0 : i32
      func.return %0 : i32
    }
  }
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main() {
      printf.print_format "Hello, I am the data in core"
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_out"} {
    func.func @print_d1() {
      printf.print_format "Hello, I am the data out core"
      func.return
    }
  }
  builtin.module attributes {kernel_type = "compute"} {
    func.func @print_compute() {
      printf.print_format "Hello, I am the compute core"
      func.return
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     printf("Hello I am the host\n");
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }

// CHECK:      void kernel_main() {
// CHECK-NEXT:     DPRINT_DATA0(DPRINT << "Hello, I am the data in core" << ENDL());
// CHECK-NEXT: }

// CHECK:      void print_d1() {
// CHECK-NEXT:     DPRINT_DATA1(DPRINT << "Hello, I am the data out core" << ENDL());
// CHECK-NEXT: }

// CHECK:      void print_compute() {
// CHECK-NEXT:     DPRINT_MATH(DPRINT << "Hello, I am the compute core" << ENDL());
// CHECK-NEXT: }
// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main() {
      printf.print_format "Hello, I am running a void data movement kernel on NOC 0."
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_out"} {
    func.func @void_dataflow1() {
      printf.print_format "Hello, I am running a void data movement kernel on NOC 1."
      func.return
    }
  }
}

// CHECK:      #include <stdint.h>
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: #include "debug/dprint.h"
// CHECK:      void kernel_main() {
// CHECK-NEXT:     DPRINT_DATA0(DPRINT << "Hello, I am running a void data movement kernel on NOC 0." << ENDL());
// CHECK-NEXT: }
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: #include "debug/dprint.h"
// CHECK:      void kernel_main() {
// CHECK-NEXT:     DPRINT_DATA1(DPRINT << "Hello, I am running a void data movement kernel on NOC 1." << ENDL());
// CHECK-NEXT: }
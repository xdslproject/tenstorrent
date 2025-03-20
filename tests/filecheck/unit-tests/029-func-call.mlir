// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host"} {
    func.func @main() -> i32 {
      %0 = arith.constant 0 : i32
      %1 = "tthost.create_device"(%0) : (i32) -> !tthost.device
      %device = memref.alloc() : memref<!tthost.device>
      memref.store %1, %device[] : memref<!tthost.device>
      %2 = arith.constant 0 : i32
      func.return %2 : i32
    }
  }
  builtin.module attributes {kernel_type = "data_in"} {
    func.func @kernel_main() {
      "dm.noc_async_read_barrier"() : () -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "data_out"} {
    func.func @data_out_api() {
      "dm.noc_async_write_barrier"() : () -> ()
      func.return
    }
  }
  builtin.module attributes {kernel_type = "compute"} {
    func.func @compute() {
      %0 = arith.constant 1 : i32
      %1 = arith.constant 2 : i32
      %2 = arith.constant 3 : i32
      %3 = arith.constant 4 : i32
      %4 = arith.constant 5 : i32
      %5 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %6 = builtin.unrealized_conversion_cast %1 : i32 to ui32
      %7 = builtin.unrealized_conversion_cast %2 : i32 to ui32
      %8 = builtin.unrealized_conversion_cast %3 : i32 to ui32
      %9 = builtin.unrealized_conversion_cast %4 : i32 to ui32
      "comp.add_tiles"(%5, %6, %7, %8, %9) : (ui32, ui32, ui32, ui32, ui32) -> ()
      func.return
    }
  }
}

// CHECK:      std::int32_t main() {
// CHECK-NEXT:     IDevice* device = CreateDevice(0);
// CHECK-NEXT:     return 0;
// CHECK-NEXT: }

// CHECK:      void kernel_main() {
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT: }

// CHECK:      void data_out_api() {
// CHECK-NEXT:     noc_async_write_barrier();
// CHECK-NEXT: }

// CHECK:      void compute() {
// CHECK-NEXT:     add_tiles(static_cast<std::uint32_t>(1), static_cast<std::uint32_t>(2), static_cast<std::uint32_t>(3), static_cast<std::uint32_t>(4), static_cast<std::uint32_t>(5));
// CHECK-NEXT: }

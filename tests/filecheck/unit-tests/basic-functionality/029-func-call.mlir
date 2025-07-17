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
      "comp.add_tiles"(%0, %1, %2, %3, %4) : (i32, i32, i32, i32, i32) -> ()
      func.return
    }
  }
}

// CHECK: CreateDevice(
// CHECK: noc_async_read_barrier(
// CHECK: noc_async_write_barrier(
// CHECK: add_tiles(

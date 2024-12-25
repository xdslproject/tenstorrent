builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 5 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %1 = memref.load %a[] : memref<i32>
    %2 = arith.constant 6 : i32
    %3 = arith.constant 3 : i32
    %4 = arith.addi %2, %3 : i32
    %5 = arith.cmpi eq, %1, %4 : i32
    scf.if %5 {
      %6 = arith.constant 2 : i32
      memref.store %6, %a[] : memref<i32>
    }
    func.return
  }
}

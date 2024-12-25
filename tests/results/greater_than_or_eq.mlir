builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 9 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %1 = memref.load %a[] : memref<i32>
    %2 = arith.constant 3 : i32
    %3 = arith.cmpi uge, %1, %2 : i32
    scf.if %3 {
      %4 = arith.constant 3 : i32
      memref.store %4, %a[] : memref<i32>
    }
    func.return
  }
}

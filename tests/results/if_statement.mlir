builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = memref.alloc() : memref<i32>
    %1 = arith.constant true
    scf.if %1 {
      %2 = arith.constant 5 : i32
      memref.store %2, %0[] : memref<i32>
    }
    func.return
  }
}

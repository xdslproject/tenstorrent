builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %a = memref.alloc() : memref<i32>
    %0 = arith.constant true
    scf.if %0 {
      %1 = arith.constant 5 : i32
      memref.store %1, %a[] : memref<i32>
    }
    func.return
  }
}

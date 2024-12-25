builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 5 : i32
    %1 = arith.constant 0 : i32
    %2 = arith.subi %1, %0 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %2, %a[] : memref<i32>
    func.return
  }
}

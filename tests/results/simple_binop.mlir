builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 2 : i32
    %1 = arith.constant 3 : i32
    %2 = arith.addi %0, %1 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %2, %a[] : memref<i32>
    func.return
  }
}

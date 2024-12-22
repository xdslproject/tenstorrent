builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 1 : i32
    %1 = memref.alloc() : memref<i32>
    memref.store %0, %1[] : memref<i32>
    %2 = arith.constant 4 : i32
    %3 = memref.alloc() : memref<i32>
    memref.store %2, %3[] : memref<i32>
    func.return
  }
}

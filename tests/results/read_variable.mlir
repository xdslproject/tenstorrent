builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 5 : i32
    %1 = memref.alloc() : memref<i32>
    memref.store %0, %1[] : memref<i32>
    %2 = memref.load %1[] : memref<i32>
    %3 = arith.constant 1 : i32
    %4 = arith.addi %2, %3 : i32
    %5 = memref.alloc() : memref<i32>
    memref.store %4, %5[] : memref<i32>
    func.return
  }
}

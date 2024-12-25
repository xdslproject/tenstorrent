builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 5 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %1 = memref.load %a[] : memref<i32>
    %2 = arith.constant 1 : i32
    %3 = arith.addi %1, %2 : i32
    %b = memref.alloc() : memref<i32>
    memref.store %3, %b[] : memref<i32>
    func.return
  }
}

builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 1 : i32
    %a = memref.alloc() : memref<i32>
    memref.store %0, %a[] : memref<i32>
    %1 = arith.constant 4 : i32
    %b = memref.alloc() : memref<i32>
    memref.store %1, %b[] : memref<i32>
    func.return
  }
}

builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = memref.alloc() : memref<i32>
    %1 = arith.constant 3 : i32
    %2 = arith.constant 5 : i32
    %3 = arith.constant 1 : i32
    %4 = memref.alloc() : memref<i32>
    scf.for %5 = %1 to %2 step %3 {
      memref.store %5, %4[] : memref<i32>
      %6 = memref.load %4[] : memref<i32>
      memref.store %6, %0[] : memref<i32>
    }
    func.return
  }
}

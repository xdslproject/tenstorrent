builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = memref.alloc() : memref<i32>
    %1 = memref.alloc() : memref<i32>
    %2 = memref.alloc() : memref<i32>
    %3 = memref.alloc() : memref<i32>
    %4 = arith.constant 3 : i32
    %5 = arith.constant 5 : i32
    %6 = arith.constant 1 : i32
    %7 = memref.alloc() : memref<i32>
    scf.for %8 = %4 to %5 step %6 {
      memref.store %8, %7[] : memref<i32>
      %9 = arith.constant 10 : i32
      memref.store %9, %0[] : memref<i32>
      %10 = arith.constant 20 : i32
      memref.store %10, %1[] : memref<i32>
      %11 = arith.constant 30 : i32
      memref.store %11, %2[] : memref<i32>
      %12 = arith.constant 40 : i32
      memref.store %12, %3[] : memref<i32>
      %13 = arith.constant 50 : i32
      memref.store %13, %0[] : memref<i32>
    }
    func.return
  }
}

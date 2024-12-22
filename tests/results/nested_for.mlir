builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = memref.alloc() : memref<i32>
    %1 = arith.constant 3 : i32
    %2 = arith.constant 5 : i32
    %3 = arith.constant 1 : i32
    %4 = memref.alloc() : memref<i32>
    scf.for %5 = %1 to %2 step %3 {
      memref.store %5, %4[] : memref<i32>
      %6 = arith.constant 7 : i32
      %7 = arith.constant 9 : i32
      %8 = arith.constant 1 : i32
      %9 = memref.alloc() : memref<i32>
      scf.for %10 = %6 to %7 step %8 {
        memref.store %10, %9[] : memref<i32>
        %11 = arith.constant 10 : i32
        memref.store %11, %0[] : memref<i32>
      }
    }
    func.return
  }
}

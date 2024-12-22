builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 0 : i32
    %1 = memref.alloc() : memref<i32>
    memref.store %0, %1[] : memref<i32>
    %2 = arith.constant 3 : i32
    %3 = arith.constant 5 : i32
    %4 = arith.constant 1 : i32
    %5 = memref.alloc() : memref<i32>
    scf.for %6 = %2 to %3 step %4 {
      memref.store %6, %5[] : memref<i32>
      %7 = memref.load %5[] : memref<i32>
      memref.store %7, %1[] : memref<i32>
    }
    func.return
  }
}

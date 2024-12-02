builtin.module {
  func.func @for_loop() {
    %0 = memref.alloc() : memref<1xi32>
    %1 = memref.alloc() : memref<1xi32>
    %2 = memref.alloc() : memref<1xi32>
    %3 = memref.alloc() : memref<1xi32>
    %4 = arith.constant 3 : i32
    %5 = arith.constant 5 : i32
    %6 = arith.constant 1 : i32
    %7 = memref.alloc() : memref<1xi32>
    scf.for %8 = %4 to %5 step %6 {
      memref.store %8, %7[] : memref<1xi32>
      %9 = arith.constant 10 : i32
      memref.store %9, %0[] : memref<1xi32>
      %10 = arith.constant 20 : i32
      memref.store %10, %1[] : memref<1xi32>
      %11 = arith.constant 30 : i32
      memref.store %11, %2[] : memref<1xi32>
      %12 = arith.constant 40 : i32
      memref.store %12, %3[] : memref<1xi32>
      %13 = arith.constant 50 : i32
      memref.store %13, %0[] : memref<1xi32>
    }
  }
}

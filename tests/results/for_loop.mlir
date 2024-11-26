builtin.module {
  func.func @for_loop() {
    %0 = memref.alloc() : memref<1xi32>
    %1 = memref.alloc() : memref<1xi32>
    %2 = memref.alloc() : memref<1xi32>
    %3 = memref.alloc() : memref<1xi32>
    %4 = arith.constant 3 : i32
    %5 = arith.constant 5 : i32
    %6 = arith.index_cast %4 : i32 to index
    %7 = arith.index_cast %5 : i32 to index
    %8 = arith.constant 1 : index
    %9 = memref.alloc() : memref<1xindex>
    scf.for %10 = %6 to %7 step %8 {
      memref.store %10, %9[] : memref<1xindex>
      %11 = arith.constant 10 : i32
      memref.store %11, %0[] : memref<1xi32>
      %12 = arith.constant 20 : i32
      memref.store %12, %1[] : memref<1xi32>
      %13 = arith.constant 30 : i32
      memref.store %13, %2[] : memref<1xi32>
      %14 = arith.constant 40 : i32
      memref.store %14, %3[] : memref<1xi32>
      %15 = arith.constant 50 : i32
      memref.store %15, %0[] : memref<1xi32>
    }
  }
}
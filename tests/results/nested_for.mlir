builtin.module {
  func.func @nested_for() {
    %0 = memref.alloc() : memref<1xi32>
    %1 = arith.constant 3 : i32
    %2 = arith.constant 5 : i32
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %5 = arith.constant 1 : index
    %6 = memref.alloc() : memref<1xindex>
    scf.for %7 = %3 to %4 step %5 {
      memref.store %7, %6[] : memref<1xindex>
      %8 = arith.constant 7 : i32
      %9 = arith.constant 9 : i32
      %10 = arith.index_cast %8 : i32 to index
      %11 = arith.index_cast %9 : i32 to index
      %12 = arith.constant 1 : index
      %13 = memref.alloc() : memref<1xindex>
      scf.for %14 = %10 to %11 step %12 {
        memref.store %14, %13[] : memref<1xindex>
        %15 = arith.constant 10 : i32
        memref.store %15, %0[] : memref<1xi32>
      }
    }
  }
}

builtin.module {
  func.func @for_loop_overwriting() {
    %0 = arith.constant 0 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.alloc() : memref<1xindex>
    memref.store %1, %2[] : memref<1xindex>
    %3 = arith.constant 3 : i32
    %4 = arith.constant 5 : i32
    %5 = arith.index_cast %3 : i32 to index
    %6 = arith.index_cast %4 : i32 to index
    %7 = arith.constant 1 : index
    %8 = memref.alloc() : memref<1xindex>
    scf.for %9 = %5 to %6 step %7 {
      memref.store %9, %8[] : memref<1xindex>
      %10 = memref.load %8[] : memref<1xindex>
      memref.store %10, %2[] : memref<1xindex>
    }
  }
}

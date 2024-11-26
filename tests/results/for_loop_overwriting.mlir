builtin.module {
  func.func @for_loop_overwriting() {
    %0 = arith.constant 0 : i32
    %1 = memref.alloc() : memref<1xindex>
    memref.store %0, %1[] : memref<1xindex>
    %2 = arith.constant 3 : i32
    %3 = arith.constant 5 : i32
    %4 = arith.index_cast %2 : i32 to index
    %5 = arith.index_cast %3 : i32 to index
    %6 = arith.constant 1 : index
    %7 = memref.alloc() : memref<1xindex>
    scf.for %8 = %4 to %5 step %6 {
      memref.store %8, %7[] : memref<1xindex>
      %9 = memref.load %7[] : memref<1xindex>
      memref.store %9, %1[] : memref<1xindex>
    }
  }
}
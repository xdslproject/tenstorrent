builtin.module {
  func.func @for_loop_use_i() {
    %0 = memref.alloc() : memref<1xindex>
    %1 = arith.constant 3 : i32
    %2 = arith.constant 5 : i32
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %5 = arith.constant 1 : index
    %6 = memref.alloc() : memref<1xindex>
    scf.for %7 = %3 to %4 step %5 {
      memref.store %7, %6[] : memref<1xindex>
      %8 = memref.load %6[] : memref<1xindex>
      memref.store %8, %0[] : memref<1xindex>
    }
  }
}

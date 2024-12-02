builtin.module {
  func.func @for_loop_overwriting() {
    %0 = arith.constant 0 : i32
    %1 = memref.alloc() : memref<1xi32>
    memref.store %0, %1[] : memref<1xi32>
    %2 = arith.constant 3 : i32
    %3 = arith.constant 5 : i32
    %4 = arith.constant 1 : i32
    %5 = memref.alloc() : memref<1xi32>
    scf.for %6 = %2 to %3 step %4 {
      memref.store %6, %5[] : memref<1xi32>
      %7 = memref.load %5[] : memref<1xi32>
      memref.store %7, %1[] : memref<1xi32>
    }
  }
}

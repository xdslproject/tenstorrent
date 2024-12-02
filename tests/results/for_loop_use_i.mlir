builtin.module {
  func.func @for_loop_use_i() {
    %0 = memref.alloc() : memref<1xi32>
    %1 = arith.constant 3 : i32
    %2 = arith.constant 5 : i32
    %3 = arith.constant 1 : i32
    %4 = memref.alloc() : memref<1xi32>
    scf.for %5 = %1 to %2 step %3 {
      memref.store %5, %4[] : memref<1xi32>
      %6 = memref.load %4[] : memref<1xi32>
      memref.store %6, %0[] : memref<1xi32>
    }
  }
}

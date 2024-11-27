builtin.module {
  func.func @evaluate_bool_nested() {
    %0 = arith.constant 5 : i32
    %1 = memref.alloc() : memref<1xi32>
    memref.store %0, %1[] : memref<1xi32>
    %2 = arith.constant 6 : i32
    %3 = arith.constant 3 : i32
    %4 = memref.load %1[] : memref<1xi32>
    %5 = arith.addi %2, %3 : i32
    %6 = arith.cmpi eq, %4, %5 : i32
    scf.if %6 {
      %7 = arith.constant 2 : i32
      memref.store %7, %1[] : memref<1xi32>
    }
  }
}

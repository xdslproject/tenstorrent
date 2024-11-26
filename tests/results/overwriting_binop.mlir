builtin.module {
  func.func @overwriting_binop() {
    %0 = arith.constant 5 : i32
    %1 = memref.alloc() : memref<1xi32>
    memref.store %0, %1[] : memref<1xi32>
    %2 = memref.load %1[] : memref<1xi32>
    %3 = arith.constant 1 : i32
    %4 = arith.addi %2, %3 : i32
    memref.store %4, %1[] : memref<1xi32>
  }
}
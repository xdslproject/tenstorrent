builtin.module {
  func.func @simple_binop() {
    %0 = arith.constant 2 : i32
    %1 = arith.constant 3 : i32
    %2 = arith.addi %0, %1 : i32
    %3 = memref.alloc() : memref<1xi32>
    memref.store %2, %3[] : memref<1xi32>
  }
}
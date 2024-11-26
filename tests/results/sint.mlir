builtin.module {
  func.func @sint() {
    %0 = arith.constant 5 : i32
    %1 = arith.constant 0 : i32
    %2 = arith.subi %1, %0 : i32
    %3 = memref.alloc() : memref<1xi32>
    memref.store %2, %3[] : memref<1xi32>
  }
}
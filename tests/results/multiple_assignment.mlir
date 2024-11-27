builtin.module {
  func.func @multiple_assignment() {
    %0 = arith.constant 1 : i32
    %1 = memref.alloc() : memref<1xi32>
    memref.store %0, %1[] : memref<1xi32>
    %2 = arith.constant 4 : i32
    %3 = memref.alloc() : memref<1xi32>
    memref.store %2, %3[] : memref<1xi32>
  }
}

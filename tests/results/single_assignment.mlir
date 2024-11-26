builtin.module {
  func.func @single_assignment() {
    %0 = arith.constant 13 : i32
    %1 = memref.alloc() : memref<1xi32>
    memref.store %0, %1[] : memref<1xi32>
  }
}
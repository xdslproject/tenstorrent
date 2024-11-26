builtin.module {
  func.func @subtraction() {
    %0 = arith.constant 4 : i32
    %1 = memref.alloc() : memref<1xi32>
    memref.store %0, %1[] : memref<1xi32>
    %2 = memref.load %1[] : memref<1xi32>
    %3 = arith.constant 10 : i32
    %4 = arith.subi %2, %3 : i32
    memref.store %4, %1[] : memref<1xi32>
  }
}
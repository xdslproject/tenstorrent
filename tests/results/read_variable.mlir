builtin.module {
  func.func @read_variable() {
    %0 = arith.constant 5 : i32
    %1 = memref.alloc() : memref<1xi32>
    memref.store %0, %1[] : memref<1xi32>
    %2 = memref.load %1[] : memref<1xi32>
    %3 = arith.constant 1 : i32
    %4 = arith.addi %2, %3 : i32
    %5 = memref.alloc() : memref<1xi32>
    memref.store %4, %5[] : memref<1xi32>
  }
}
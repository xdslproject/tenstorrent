builtin.module {
  func.func @nested_binops() {
    %0 = arith.constant 2 : i32
    %1 = arith.constant 3 : i32
    %2 = arith.constant 1 : i32
    %3 = arith.muli %0, %1 : i32
    %4 = arith.addi %2, %3 : i32
    %5 = arith.constant 4 : i32
    %6 = arith.addi %4, %5 : i32
    %7 = memref.alloc() : memref<1xi32>
    memref.store %6, %7[] : memref<1xi32>
  }
}

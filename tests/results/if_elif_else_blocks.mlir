builtin.module {
  func.func @if_elif_else_blocks() {
    %0 = arith.constant 5 : i32
    %1 = memref.alloc() : memref<1xi32>
    memref.store %0, %1[] : memref<1xi32>
    %2 = memref.alloc() : memref<1xi32>
    %3 = memref.load %1[] : memref<1xi32>
    %4 = arith.constant 7 : i32
    %5 = arith.cmpi eq, %3, %4 : i32
    scf.if %5 {
      %6 = arith.constant 2 : i32
      memref.store %6, %1[] : memref<1xi32>
      %7 = arith.constant 3 : i32
      memref.store %7, %2[] : memref<1xi32>
    } else {
      %8 = memref.load %1[] : memref<1xi32>
      %9 = arith.constant 5 : i32
      %10 = arith.cmpi eq, %8, %9 : i32
      scf.if %10 {
        %11 = arith.constant 1 : i32
        memref.store %11, %1[] : memref<1xi32>
        %12 = arith.constant 2 : i32
        memref.store %12, %2[] : memref<1xi32>
      } else {
        %13 = arith.constant 10 : i32
        memref.store %13, %1[] : memref<1xi32>
        %14 = arith.constant 9 : i32
        memref.store %14, %2[] : memref<1xi32>
      }
    }
  }
}

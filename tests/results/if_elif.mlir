builtin.module {
  func.func @if_elif() {
    %0 = arith.constant 5 : i32
    %1 = memref.alloc() : memref<1xi32>
    memref.store %0, %1[] : memref<1xi32>
    %2 = memref.load %1[] : memref<1xi32>
    %3 = arith.constant 7 : i32
    %4 = arith.cmpi eq, %2, %3 : i32
    scf.if %4 {
      %5 = arith.constant 2 : i32
      memref.store %5, %1[] : memref<1xi32>
    } else {
      %6 = memref.load %1[] : memref<1xi32>
      %7 = arith.constant 5 : i32
      %8 = arith.cmpi eq, %6, %7 : i32
      scf.if %8 {
        %9 = arith.constant 1 : i32
        memref.store %9, %1[] : memref<1xi32>
      }
    }
  }
}

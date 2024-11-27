builtin.module {
  func.func @bool_assign() {
    %0 = arith.constant true
    %1 = memref.alloc() : memref<1xi1>
    memref.store %0, %1[] : memref<1xi1>
    %2 = arith.constant false
    %3 = memref.alloc() : memref<1xi1>
    memref.store %2, %3[] : memref<1xi1>
    %4 = memref.load %1[] : memref<1xi1>
    %5 = memref.load %3[] : memref<1xi1>
    %6 = arith.andi %4, %5 : i1
    %7 = memref.alloc() : memref<1xi1>
    memref.store %6, %7[] : memref<1xi1>
    %8 = memref.load %7[] : memref<1xi1>
    scf.if %8 {
      %9 = arith.constant false
      memref.store %9, %1[] : memref<1xi1>
    }
    %10 = arith.constant 5 : i32
    %11 = memref.alloc() : memref<1xi32>
    memref.store %10, %11[] : memref<1xi32>
    %12 = memref.load %11[] : memref<1xi32>
    %13 = arith.constant 10 : i32
    %14 = arith.cmpi ult, %12, %13 : i32
    %15 = memref.load %7[] : memref<1xi1>
    %16 = arith.ori %14, %15 : i1
    %17 = memref.alloc() : memref<1xi1>
    memref.store %16, %17[] : memref<1xi1>
  }
}

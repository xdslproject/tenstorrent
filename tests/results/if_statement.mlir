builtin.module {
  func.func @if_statement() {
    %0 = memref.alloc() : memref<1xi32>
    %1 = arith.constant true
    scf.if %1 {
      %2 = arith.constant 5 : i32
      memref.store %2, %0[] : memref<1xi32>
    }
  }
}
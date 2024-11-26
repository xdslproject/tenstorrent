builtin.module {
  func.func @floating_point() {
    %0 = arith.constant 2.730000e+01 : f32
    %1 = arith.constant 4.120000e+01 : f32
    %2 = arith.addf %0, %1 : f32
    %3 = memref.alloc() : memref<1xf32>
    memref.store %2, %3[] : memref<1xf32>
    %4 = arith.constant 1.620000e+01 : f32
    %5 = arith.constant 1.310000e+01 : f32
    %6 = arith.mulf %4, %5 : f32
    %7 = memref.alloc() : memref<1xf32>
    memref.store %6, %7[] : memref<1xf32>
  }
}
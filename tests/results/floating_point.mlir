builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 2.730000e+01 : f32
    %1 = arith.constant 4.120000e+01 : f32
    %2 = arith.addf %0, %1 : f32
    %3 = memref.alloc() : memref<f32>
    memref.store %2, %3[] : memref<f32>
    %4 = arith.constant 1.620000e+01 : f32
    %5 = arith.constant 1.310000e+01 : f32
    %6 = arith.mulf %4, %5 : f32
    %7 = memref.alloc() : memref<f32>
    memref.store %6, %7[] : memref<f32>
    func.return
  }
}

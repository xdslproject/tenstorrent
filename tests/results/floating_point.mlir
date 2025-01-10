builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 2.730000e+01 : f32
    %1 = arith.constant 4.120000e+01 : f32
    %2 = arith.addf %0, %1 : f32
    %a = memref.alloc() : memref<f32>
    memref.store %2, %a[] : memref<f32>
    %3 = arith.constant 1.620000e+01 : f32
    %4 = arith.constant 1.310000e+01 : f32
    %5 = arith.mulf %3, %4 : f32
    %b = memref.alloc() : memref<f32>
    memref.store %5, %b[] : memref<f32>
    func.return
  }
}

builtin.module attributes  {"kernel_type" = "data_in"} {
  func.func @kernel_main() {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 1 : i32
    "cb.push_back"(%0, %1) : (i32, i32) -> ()
    func.return
  }
}

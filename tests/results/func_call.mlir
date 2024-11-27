builtin.module {
  func.func @func_call() {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 1 : i32
    "cb.push_back"(%0, %1) : (i32, i32) -> ()
  }
}
